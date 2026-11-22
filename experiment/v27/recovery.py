"""Inject a slow packet into isolated worker processes; keep all customer services untouched."""
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile
from .data import ROOT,OUT

def controlled(conn):
    time.sleep(.7)
    conn.send('ready')
    try:
        while True:
            packet,_,_=conn.recv()
            if packet['label']=='slow':time.sleep(4)
            conn.send((True,{'probe':packet['label']}))
    except (OSError,EOFError):
        pass

def sequence(module):
    w=module.GuardWorker(target=controlled)
    try:
        w.warm()
        seen=[]
        for label in ('slow','next'):
            try:
                seen.append(w.check({'label':label},None,None,time.monotonic()+.2)['probe'])
            except module.BudgetExceeded:
                seen.append('timeout')
        return seen
    finally:w.close()

def main():
    from app import contest_worker
    with tempfile.TemporaryDirectory() as d,ZipFile(OUT/'v26-runtime.zip') as z:
        old=Path(d)/'worker.py'
        old.write_bytes(z.read('app/contest_worker.py'))
        spec=importlib.util.spec_from_file_location('historical_guard',old)
        previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
        stats={'v26':sequence(previous),'v27':sequence(contest_worker)}
    assert stats=={'v26':['timeout','timeout'],'v27':['timeout','next']},stats
    (OUT/'recovery.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print(stats)

if __name__=='__main__':main()
