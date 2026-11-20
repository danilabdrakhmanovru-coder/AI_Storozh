# Проверяет адаптер Ollama при сбоях связи и повреждённых ответах, включая восстановление после ошибки.
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.model import Ollama


def response(content, finish='stop'):
    payload = {'message': {'content': content}, 'done_reason': finish}
    return io.BytesIO(json.dumps(payload).encode())


class OllamaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.model = Ollama()
        self.model.enabled = True

    def test_timeout_releases_slot_for_next_request(self):
        good = response('{"risk":0.1,"reason":"ok"}')
        with patch('urllib.request.urlopen', side_effect=[TimeoutError(), good]):
            self.assertEqual(self.model.judge({'text':'Первый'})['status'], 'error')
            self.assertEqual(self.model.judge({'text':'Следующий'})['risk'], 0.1)

    def test_incomplete_or_malformed_answer_has_no_risk_score(self):
        for content, finish in [('{"risk":0,"reason":"ok"}','length'),('broken','stop'),('{"risk":-1,"reason":"ok"}','stop')]:
            with self.subTest(content=content, finish=finish), patch('urllib.request.urlopen', return_value=response(content, finish)):
                result = self.model.judge({'text':'Привет'})
                self.assertEqual(result['status'], 'error')
                self.assertNotIn('risk', result)

    def test_disabled_model_makes_no_network_call(self):
        self.model.enabled = False
        with patch('urllib.request.urlopen') as network:
            self.assertEqual(self.model.judge({'text':'Привет'})['status'], 'disabled')
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
