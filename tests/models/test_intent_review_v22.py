# Проверяет локальные веса v22, сохранение полномочий и границы пересмотра решения.
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from app.contest import ContestRequest,inspect_request,read_policy,parse_policy
from app.intent_review import reviewer


class IntentReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=read_policy()
        cls.limits=parse_policy(cls.raw)['thresholds']

    def check_request(self,text,role='developer',**context):
        req=ContestRequest(user_prompt=text,context={'user_role':role,'target_agent':'it_ops_agent',**context})
        return inspect_request(req,self.limits,self.raw)

    def test_weights_and_scores_are_local_and_stateless(self):
        model=reviewer()
        self.assertEqual(model.features.detector.session.get_providers(),['CPUExecutionProvider'])
        first=model.review('When does the library open?')
        model.review('What colours appear on a map?')
        self.assertEqual(first,model.review('When does the library open?'))
        self.assertEqual(len(first),3)

    def test_cannot_override_policy_or_document_path(self):
        with patch.dict(os.environ,STOROZH_ATTACHMENT_INTENT='9',STOROZH_VARIANT='B',STOROZH_ENCODER_DIR=''):
            with patch('app.intent_review.reviewer',side_effect=AssertionError('Policy or document reached direct reviewer')):
                self.assertEqual(self.check_request('Describe a tulip.','unknown-role')['decision'],'block')
                self.assertEqual(self.check_request('Перезапусти payments-api в production.')['decision'],'block')
                self.check_request('Summarize the note.',attachments=[dict(content='Small yellow flowers.')])
                self.check_request('Describe a tulip.',history=[dict(role='assistant',content='The catalogue is ready.')])

    def test_only_categories_authorized_by_policy_can_block(self):
        with patch.dict(os.environ,STOROZH_ATTACHMENT_INTENT='9',STOROZH_VARIANT='B',STOROZH_ENCODER_DIR=''):
            with patch('app.intent_review.reviewer') as mock:
                mock.return_value.review.return_value=(False,'out_of_policy',False)
                self.assertEqual(self.check_request('What colour is a ripe pear?')['decision'],'allow')
                mock.return_value.review.return_value=(False,'prompt_injection',False)
                hit=self.check_request('What colour is a ripe pear?')
                self.assertEqual(hit['decision'],'block')
                self.assertEqual(hit['evidence'][0]['rule'],'intent_review_v22')


if __name__=='__main__':unittest.main()
