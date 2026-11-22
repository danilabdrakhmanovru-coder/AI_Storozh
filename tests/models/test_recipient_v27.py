# Проверяет охват вложения, ограничение числа окон и сохранение неопределённости при неполном чтении.
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app import recipient_fast as model
from app.recipient_model import encode


class CoverageTests(unittest.TestCase):
    def test_reference_document_cannot_clear_a_policy_deny_on_the_prompt(self):
        from app.contest import ContestRequest,inspect_request,parse_policy,read_policy
        policy=yaml.safe_load(read_policy())
        policy['agents']['it_ops_agent']['forbidden_actions'].append(dict(
            name='limited_summary',description='Проверка запрета из YAML',patterns=['^summarize the attached document']))
        raw=yaml.safe_dump(policy,allow_unicode=True).encode()
        req=ContestRequest(user_prompt='Summarize the attached document.',
            context=dict(user_role='developer',target_agent='it_ops_agent',attachments=[{'content':'A catalogue of chairs.'}]))
        with patch.dict(os.environ,STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='13',STOROZH_ENCODER_DIR=''), \
                patch('app.recipient_calibration.classify',return_value='reference'):
            got=inspect_request(req,parse_policy(raw)['thresholds'],raw)
        self.assertEqual(got['decision'],'block')
        self.assertIn('limited_summary',got['explanation'])

    def test_local_ranker_reads_every_character_and_exact_tail(self):
        for pos in (0,1000,1190):
            body='x'*pos+'Q'+'x'*(1190-pos)
            rank=Mock()
            rank.decision_function.side_effect=lambda rows:np.array([int('Q' in row) for row in rows])
            with self.subTest(pos=pos),patch.object(model,'ranker',return_value=rank):
                full,local,_=model.lexical(body)
                self.assertEqual((full,local),(1,1))
                self.assertTrue(body.endswith(rank.decision_function.call_args.args[0][-1]))

    def test_partial_coverage_or_strong_local_signal_prevents_clear(self):
        head=Mock()
        head.predict_proba.return_value=np.array([[.95,.05]])
        cfg=dict(head=head,upper=.8,lower=.4,max_reference_span=.7)
        z=np.zeros(8)
        self.assertEqual(model.decision(cfg,z,True),'reference')
        self.assertEqual(model.decision(cfg,z,False),'uncovered')
        z[6]=1.
        self.assertEqual(model.decision(cfg,z,True),'uncertain')
        head.predict_proba.return_value=np.array([[.05,.95]])
        self.assertEqual(model.decision(cfg,z,False),'agent')

    def test_bounded_encoder_retains_head_and_tail(self):
        tok=Mock()
        tok.encode.return_value=SimpleNamespace(ids=[1,91,*([8]*1200),92,2])
        tok.token_to_id.return_value=0
        detector=Mock()
        detector.run.return_value=np.zeros((3,2)),np.zeros((3,5))
        pooled,complete,review=encode(detector,tok,'document',window=258,stride=210,max_windows=3,with_review=True)
        self.assertEqual(detector.run.call_count,1)
        np.testing.assert_array_equal(review[5:],pooled[3:])
        self.assertAlmostEqual(review[4],np.log1p(1202))
        batch=detector.run.call_args.args[1]['input_ids']
        self.assertFalse(complete)
        self.assertEqual(batch.shape,(3,258))
        self.assertIn(91,batch[0]);self.assertIn(92,batch[-1])

    def test_fallback_uses_features_from_the_same_document_only(self):
        from app import document_review
        req=SimpleNamespace(context=SimpleNamespace(history=[],attachments=[SimpleNamespace(content='catalogue')]))
        head=Mock()
        head.predict_proba.return_value=np.array([[.05,.95]])
        cfg=dict(classifier=head,block=.8)
        verdict=dict(decision='allow',risk_category='none')
        policy=dict(always_block=['indirect_injection'])
        with patch.object(document_review,'model',return_value=cfg),patch.object(document_review,'probability',return_value=.1) as encode_again:
            vector=np.arange(9)
            got=document_review.review(req,verdict,policy,Mock(),review_vectors={'catalogue':vector})
            self.assertEqual(got['decision'],'block')
            np.testing.assert_array_equal(head.predict_proba.call_args.args[0],vector[None,:])
            encode_again.assert_not_called()
            self.assertEqual(document_review.review(req,verdict,policy,Mock(),review_vectors={'another':vector}),verdict)
            encode_again.assert_called_once_with('catalogue')

if __name__=='__main__':unittest.main()
