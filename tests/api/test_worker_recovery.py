# Проверяет ответ следующему HTTP-запросу после тайм-аута без повторного ручного прогрева.
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.contest_worker import BudgetExceeded,GuardWorker


def delayed(conn):
    time.sleep(.6)
    conn.send('ready')
    try:
        while True:
            packet,_,_=conn.recv()
            if packet['user_prompt']=='slow':time.sleep(6)
            conn.send((True,dict(decision='allow',risk_level='none',risk_category='none',
                                explanation=packet['user_prompt'])))
    except (OSError, EOFError):
        pass


class WorkerRecovery(unittest.TestCase):
    def test_short_waiters_do_not_reset_the_cold_start(self):
        w=GuardWorker(target=delayed)
        try:
            w.warm()
            for _ in range(2):
                with self.assertRaises(BudgetExceeded):
                    w.check({'user_prompt':'slow'},None,None,time.monotonic()+.08)
            until=time.monotonic()+12
            while time.monotonic()<until:
                try:
                    result=w.check({'user_prompt':'recovered'},None,None,time.monotonic()+.08)
                except BudgetExceeded:
                    continue
                self.assertEqual(result['explanation'],'recovered')
                break
            else:self.fail('Short polling requests prevented recovery')
        finally:w.close()

    def test_next_http_check_uses_ready_process_and_its_own_result(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.storage import Store
        w=GuardWorker(target=delayed)
        store=Store(':memory:')
        body={'context':{'user_role':'developer','target_agent':'it_ops_agent'}}
        try:
            with patch.dict(os.environ,STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='13',STOROZH_API_KEY='',STOROZH_CLIENTS_JSON='{}'), \
                    patch('app.main.cpu_worker',w),patch('app.main.store',store), \
                    patch('app.main.BUDGET_SECONDS',.25),TestClient(app) as client:
                first=client.post('/v1/check',json={**body,'user_prompt':'slow'}).json()
                self.assertTrue(first['check_error'])
                self.assertEqual(first['decision'],'block')
                second=client.post('/v1/check',json={**body,'user_prompt':'fresh'}).json()
                self.assertEqual((second['decision'],second['explanation']),('allow','fresh'))
        finally:
            w.close()
            store.db.close()


if __name__=='__main__':unittest.main()
