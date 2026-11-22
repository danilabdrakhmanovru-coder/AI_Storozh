# Проверяет границы полномочий смысловой модели, источник сигнала и автономную загрузку весов v21.
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from app.contest import ContestRequest,inspect_request,read_policy,parse_policy
from app.semantic_review import reviewer

class SemanticReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=read_policy()
        cls.thresholds=parse_policy(cls.raw)['thresholds']

    def decide(self,text,role='developer',**extra):
        req=ContestRequest(user_prompt=text,context=dict(user_role=role,target_agent='it_ops_agent',**extra))
        return inspect_request(req,self.thresholds,self.raw)

    def test_local_weights_are_batch_independent_and_explainable(self):
        model=reviewer()
        first=model.assess('What colour is a ripe pear?')
        model.assess('Which character narrates the second chapter?')
        self.assertEqual(first,model.assess('What colour is a ripe pear?'))
        self.assertIsInstance(first[0],bool)

    def test_policy_and_unknown_roles_precede_review(self):
        with patch.dict(os.environ,STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='8',STOROZH_ENCODER_DIR=''), \
             patch('app.semantic_review.reviewer',side_effect=AssertionError('Unexpected semantic decision')):
            self.assertEqual(self.decide('Describe a fountain.',role='absent-role')['decision'],'block')
            self.assertEqual(self.decide('Перезапусти payments-api в production.')['decision'],'block')

    def test_review_is_scoped_to_policy_and_plain_text(self):
        with patch.dict(os.environ,STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='8',STOROZH_ENCODER_DIR=''), \
             patch('app.semantic_review.reviewer') as model:
            model.return_value.assess.return_value=(False,'jailbreak')
            req='What colour is a ripe pear?'
            hit=self.decide(req)
            self.assertEqual((hit['decision'],hit['risk_category']),('block','jailbreak'))
            self.assertEqual(hit['evidence'][0]['source'],'prompt')
            model.return_value.assess.return_value=(False,'out_of_policy')
            self.assertEqual(self.decide(req)['decision'],'allow')
            model.reset_mock()
            self.decide(req,attachments=[dict(content='The pear is green.')])
            model.assert_not_called()

    def test_can_only_clear_learned_injection_votes(self):
        with patch.dict(os.environ,STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='8',STOROZH_ENCODER_DIR=''), \
             patch('app.semantic_review.reviewer') as model,patch('app.contest.predict',return_value='harmful_content'):
            model.return_value.assess.return_value=(True,None)
            out=self.decide('Describe a pear.')
            self.assertEqual((out['decision'],out['risk_category']),('block','harmful_content'))

if __name__=='__main__':unittest.main()

