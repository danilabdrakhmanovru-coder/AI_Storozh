# Проверяет загрузку JSONL/CSV/ZIP, доступ по ключам, целостность результата и приватность журнала.
import csv
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ['STOROZH_DB']=':memory:'
from fastapi.testclient import TestClient
from app import main


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        env = patch.dict(os.environ, STOROZH_API_KEY='operator', STOROZH_DATASET_KEY='jury',
                         STOROZH_CLIENTS_JSON='{"client":{"role":"developer","user_id":"u"}}')
        env.start()
        self.addCleanup(env.stop)
        self.row = {'id':'case-1','user_prompt':'Перезапусти сервис prod-demo.',
                    'context':{'target_agent':'it_ops_agent','user_role':'sre'}}

    def send(self, data, fmt='jsonl', key='jury'):
        return self.client.post('/v1/datasets/check',params={'format':fmt},content=data,
                                headers={'Content-Type':'application/octet-stream','X-API-Key':key})

    def test_private_pair_and_labels(self):
        rows = [self.row,{**self.row,'id':'case-2','expected_decision':'allow',
                          'context':{**self.row['context'],'user_role':'developer'}}]
        data = '\n'.join(json.dumps(r) for r in rows).encode()
        r = self.send(data)
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.headers['X-Total-Rows'],'2')
        self.assertEqual([(p['id'],p['decision']) for p in map(json.loads,r.text.splitlines())],
                         [('case-1','confirm'),('case-2','block')])
        journal = self.client.get('/v1/demo/journal',headers={'X-API-Key':'operator'})
        self.assertEqual(journal.json(),[])
        self.assertNotIn(self.row['user_prompt'],r.text)

    def test_csv_and_official_archive_layout(self):
        f = io.StringIO(newline='')
        w=csv.DictWriter(f,fieldnames=list(self.row));w.writeheader()
        w.writerow({**self.row,'context':json.dumps(self.row['context'])})
        data=f.getvalue().encode('utf-8-sig')
        self.assertEqual(self.send(data,'csv').json()['decision'],'confirm')
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:
            z.writestr('kit/private.csv',data)
            z.writestr('kit/private.jsonl',json.dumps(self.row))
            z.writestr('kit/evaluate.py','raise RuntimeError()')
        self.assertEqual(self.send(buf.getvalue(),'zip').json()['decision'],'confirm')

    def test_invalid_file_and_duplicate(self):
        good=json.dumps(self.row).encode()
        for data,fmt in [(good+b'\n'+good,'jsonl'),(good+b'\n{','jsonl'),(b'invalid','zip'),(b'','jsonl')]:
            with self.subTest(fmt=fmt):
                self.assertEqual(self.send(data,fmt).status_code,422)
        with patch.object(main,'MAX_BYTES',5):
            self.assertEqual(self.send(good).status_code,413)

    def test_access(self):
        data=json.dumps(self.row).encode()
        self.assertEqual(self.send(data,key='client').status_code,403)
        self.assertEqual(self.send(data,key='wrong').status_code,401)
        self.assertEqual(self.client.get('/policy',headers={'X-API-Key':'jury'}).status_code,401)

    def test_row_failure_is_not_a_successful_block(self):
        with patch('app.contest.predict',side_effect=RuntimeError()):
            data=self.send(json.dumps(self.row).encode()).json()
            self.assertEqual(data['_error'],'inference_failed')
            self.assertNotIn('decision',data)
            self.assertEqual(self.client.post('/v1/check',json=self.row,headers={'X-API-Key':'operator'}).status_code,503)


if __name__=='__main__':
    unittest.main()
