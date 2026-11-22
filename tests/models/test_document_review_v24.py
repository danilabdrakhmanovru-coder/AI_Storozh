# Проверяет границы полномочий модели вложений и повторную проверку исходной команды.
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.contest import ContestRequest
from app.document_review import review


class DocumentReviewTests(unittest.TestCase):
    policy = {'always_block': {'indirect_injection', 'prompt_injection'}}

    def request(self, docs=1, history=None):
        context = dict(target_agent='it_ops_agent', user_role='developer', history=history or [],
                       attachments=[dict(content='Quarterly notes.') for _ in range(docs)])
        return ContestRequest(user_prompt='Summarize the record.', context=context)

    def test_only_learned_attachment_verdict_can_be_relaxed(self):
        recheck = Mock()
        with patch('app.document_review.probability', side_effect=AssertionError('policy must prevail')):
            for source, rule in [('attachment','policy'), ('prompt','attachment_passage')]:
                verdict = dict(decision='block', risk_category='indirect_injection',
                               evidence=[dict(source=source, rule=rule)])
                self.assertIs(review(self.request(),verdict,self.policy,recheck),verdict)
        recheck.assert_not_called()

    def test_soft_verdict_rechecks_original_prompt_authority(self):
        verdict = dict(decision='block', risk_category='indirect_injection',
                       evidence=[dict(source='attachment',rule='attachment_passage')])
        policy_verdict = dict(decision='block',risk_category='tool_abuse')
        recheck = Mock(return_value=policy_verdict)
        with patch('app.document_review.model',return_value={'clear':.1}), patch('app.document_review.probability',return_value=.01):
            self.assertEqual(review(self.request(),verdict,self.policy,recheck),policy_verdict)
        recheck.assert_called_once_with()

    def test_new_block_records_attachment_and_preserves_confirm(self):
        with patch('app.document_review.model',return_value={'block':.8}), patch('app.document_review.probability',return_value=.99):
            res = review(self.request(),{'decision':'allow'},self.policy,Mock())
            self.assertEqual((res['decision'],res['risk_category']),('block','indirect_injection'))
            self.assertEqual(res['evidence'][0]['source'],'attachment')
            verdict = {'decision':'confirm'}
            self.assertIs(review(self.request(),verdict,self.policy,Mock()),verdict)

    def test_unsupported_contexts_retain_incumbent(self):
        verdict = {'decision':'allow'}
        with patch('app.document_review.probability',side_effect=AssertionError('outside trained scope')):
            for req in [self.request(docs=0),self.request(docs=2),self.request(history=[dict(role='user',content='Hi')])]:
                self.assertIs(review(req,verdict,self.policy,Mock()),verdict)
            req = self.request()
            req.context.attachments[0].content = 'x' * 12001
            self.assertIs(review(req,verdict,self.policy,Mock()),verdict)


if __name__=='__main__': unittest.main()
