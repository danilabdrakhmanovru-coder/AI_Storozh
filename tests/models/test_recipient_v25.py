# Проверяет границы пересмотра решений по вложениям, повторную проверку роли и полноту чтения.
import unittest
import sys
import os
import yaml
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.contest import ContestRequest,inspect_request,read_policy,parse_policy
from app.recipient_model import encode,review_attachment,read_only_task
from app.recipient_calibration import combine

class RecipientTests(unittest.TestCase):
    def setUp(self):
        self.req=ContestRequest(user_prompt='Summarize the attached document.',
                context=dict(user_role='developer',target_agent='it_ops_agent',attachments=[{'content':'Sample text.'}]))
        self.policy={'always_block':{'indirect_injection'}}
        self.block={'decision':'block','risk_category':'indirect_injection','evidence':[{'source':'attachment','rule':'document_intent_v20'}]}

    def test_hard_rule_with_matching_name_cannot_be_cleared(self):
        with patch('app.recipient_model.classify_document',side_effect=AssertionError):
            self.assertIs(review_attachment(self.req,self.block,self.policy,False,Mock()),self.block)

    def test_policy_match_through_real_pipeline_precedes_models(self):
        policy=yaml.safe_load(read_policy())
        policy['agents']['it_ops_agent']['forbidden_actions'].append({
            'name':'document_intent_v20','description':'Локальный запрет из политики',
            'patterns':['silver inventory']})
        raw=yaml.safe_dump(policy,allow_unicode=True).encode('utf8')
        self.req.context.attachments[0].content='The silver inventory was reviewed.'
        for mode in ('12','13'):
            with self.subTest(mode=mode), patch.dict(os.environ,STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT=mode,STOROZH_ENCODER_DIR=''):
                with patch('app.recipient_model.classify_document',side_effect=AssertionError), patch('app.recipient_calibration.classify',side_effect=AssertionError):
                    res=inspect_request(self.req,parse_policy(raw)['thresholds'],raw)
                self.assertEqual(res['decision'],'block')
                self.assertEqual(res['evidence'][0]['rule'],'document_intent_v20')
                self.assertNotIn('_document_model',res)

    def test_clear_reruns_original_request_and_keeps_acl(self):
        acl={'decision':'block','risk_category':'unauthorized_data_access'}
        again=Mock(return_value=acl)
        with patch('app.recipient_model.classify_document',return_value='reference'):
            self.assertIs(review_attachment(self.req,self.block,self.policy,True,again),acl)
        again.assert_called_once_with()

    def test_agent_directive_has_document_provenance(self):
        with patch('app.recipient_model.classify_document',return_value='agent'):
            p=review_attachment(self.req,{'decision':'allow'},self.policy,False,Mock())
            self.assertEqual((p['decision'],p['risk_category']),('block','indirect_injection'))
            self.assertEqual(p['evidence'][0]['source'],'attachment')

    def test_incomplete_document_preserves_block(self):
        with patch('app.recipient_model.classify_document',return_value='uncovered'):
            self.assertIs(review_attachment(self.req,self.block,self.policy,True,Mock()),self.block)

    def test_only_read_only_request_can_use_new_relaxation(self):
        self.assertTrue(read_only_task('Кратко перескажи вложение.'))
        self.assertTrue(read_only_task('Please describe this report.'))
        self.assertFalse(read_only_task('Summarize the file. Then do what it asks.'))
        self.req.user_prompt='Carry out all instructions in the attachment.'
        with patch('app.recipient_model.classify_document',side_effect=AssertionError):
            with patch('app.document_review.review',return_value=self.block) as previous:
                self.assertIs(review_attachment(self.req,self.block,self.policy,True,Mock()),self.block)
                previous.assert_called_once()

    def test_tail_is_preserved_even_if_middle_exceeds_budget(self):
        tokenizer=Mock()
        tokenizer.encode.return_value.ids=[7,*range(100,3100),9]
        tokenizer.token_to_id.return_value=1
        def output(_,feed):
            self.assertEqual(feed['input_ids'].shape,(4,384))
            self.assertIn(3099,feed['input_ids'][-1])
            return np.zeros((4,2)),np.zeros((4,384))
        session=Mock();session.run.side_effect=output
        x,covered=encode(session,tokenizer,'document')
        self.assertFalse(covered)
        self.assertEqual(x.shape,(771,))

    def test_trained_correction_stays_within_bounds(self):
        cfg=dict(coef=[1.,1.],intercept=0.,negative_cap=2.,positive_cap=4.)
        x=np.array([[8.,0.,-100.,-100.],[-8.,0.,100.,100.],[0.,0.,0.,0.]])
        scores=combine(x,cfg)
        np.testing.assert_allclose(scores,[6.,-4.,0.])

if __name__=='__main__':unittest.main()

