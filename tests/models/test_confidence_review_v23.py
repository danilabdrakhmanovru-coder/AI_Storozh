# Проверяет, что v23 пересматривает только решения классификаторов и сохраняет запреты политики.
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from app.contest import ContestRequest, inspect_request, parse_policy, read_policy
from app.confidence_review import reviewer


class ConfidenceTests(unittest.TestCase):
    def check(self,text,role='developer',**context):
        raw=read_policy()
        r=ContestRequest(user_prompt=text,context={'target_agent':'it_ops_agent','user_role':role,**context})
        return inspect_request(r,parse_policy(raw)['thresholds'],raw)

    def test_authority_and_untrusted_sources_never_reach_reviewer(self):
        with patch.dict(os.environ,STOROZH_ATTACHMENT_INTENT='10',STOROZH_VARIANT='B',STOROZH_ENCODER_DIR=''):
            with patch('app.confidence_review.reviewer',side_effect=AssertionError('unexpected model override')):
                self.assertEqual(self.check('Расскажи про карту.','unknown')['decision'],'block')
                self.assertEqual(self.check('Перезапусти payments-api в production.')['decision'],'block')
                self.check('Summarize the note.',attachments=[dict(content='There are two items.')])
                self.check('How many?',history=[dict(role='assistant',content='There are two items.')])

    def test_abstention_retains_actual_decision_without_internal_flags(self):
        with patch.dict(os.environ,STOROZH_ATTACHMENT_INTENT='10',STOROZH_VARIANT='B',STOROZH_ENCODER_DIR=''):
            with patch('app.confidence_review.reviewer') as mock:
                mock.return_value.revise.return_value=None
                result=self.check('What colour is a ripe pear?')
                self.assertEqual(result['decision'],'allow')
                self.assertNotIn('_text_review',result)
                prior=mock.return_value.revise.call_args.args[1]
                self.assertEqual(prior['decision'],'allow')

    def test_loaded_model_uses_every_input_dimension(self):
        model=reviewer()
        request=ContestRequest(user_prompt='When will the library open?',context={'target_agent':'it_ops_agent','user_role':'developer'})
        old={'decision':'allow','risk_category':'none'}
        a=model.revise(request,old,{'prompt_injection'})
        b=model.revise(request,old,{'prompt_injection'})
        self.assertEqual(a,b)
        self.assertIn(a and a['risk_category'],(None,'prompt_injection'))


if __name__=='__main__': unittest.main()
