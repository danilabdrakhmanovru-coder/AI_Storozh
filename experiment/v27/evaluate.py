"""Sequential requests to the standard FastAPI startup and bounded worker. No retries or role overrides."""
import argparse
import importlib.util
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import numpy as np
import psutil
from .data import OUT,ROOT,read,cases,release_digest

spec=importlib.util.spec_from_file_location('astra_evaluator',ROOT/'artifacts/astra_public/evaluate.py')
ev=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
CTX=dict(user_role='developer',target_agent='it_ops_agent',history=[],attachments=[])

def reason(p):
    evidence=p.get('evidence') or []
    return evidence[0].get('rule','rule') if evidence else None

@contextmanager
def live():
    env={k:v for k,v in os.environ.items() if not k.startswith('STOROZH_')}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        env.update(STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='13',STOROZH_DB=str(Path(d)/'audit.db'),
                   ASTRA_POLICY_PATH=str(ROOT/'security_policy.yaml'),PYTHONPATH=str(ROOT),
                   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='2')
        cmd=[sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(port),
             '--no-access-log','--log-level','warning']
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env)
        url=f'http://127.0.0.1:{port}'
        start=time.monotonic()
        try:
            while time.monotonic()-start<260:
                if proc.poll() is not None:raise RuntimeError('Startup failed')
                try:
                    with urllib.request.urlopen(url+'/health',timeout=.4):break
                except OSError:time.sleep(.2)
            else:raise TimeoutError('Startup')
            yield url+'/v1/check',time.monotonic()-start,psutil.Process(proc.pid)
        finally:
            if proc.poll() is None:
                children=psutil.Process(proc.pid).children(recursive=True)
                proc.terminate()
                for child in reversed(children):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                psutil.wait_procs(children,timeout=4)
                proc.wait(timeout=10)

def inputs(name):
    if name=='public':
        return [('public',r['expected_decision'],r) for r in ev.load_jsonl(ROOT/'artifacts/astra_public/dev.jsonl')]
    return [(s,'block' if y else 'allow',dict(id=str(i),user_prompt='Summarize the attached document.',
              context={**CTX,'attachments':[dict(type='document',name='record.txt',content=t)]}))
            for i,(s,t,y) in enumerate(cases(name))]

def evaluate(tag,suites):
    from app.recipient_fast import calibration
    calibration()  # Ensure the weights belong to the installed encoder before startup.
    if 'holdout' in suites:
        for k,v in read(OUT/'candidate.json').items():
            if release_digest(ROOT/k)!=v:raise ValueError('Candidate changed: '+k)
    paths=[OUT/f'{tag}-{s}.json' for s in suites]
    if any(p.exists() for p in paths):raise FileExistsError(paths)
    proc=psutil.Process()
    proc.cpu_affinity(sorted(proc.cpu_affinity())[-8:])
    os.environ['no_proxy']='localhost,127.0.0.1'
    with live() as (url,startup,server):
        warm=dict(id='warm',user_prompt='Summarize the attached document.',
                  context={**CTX,'attachments':[dict(content='The inventory has two rows.')]})
        for _ in range(5):
            p=ev.call_endpoint(url,warm)
            assert ev.valid_prediction(p) and not p.get('check_error'),p.get('explanation')
        for name,path in zip(suites,paths):
            examples=inputs(name)
            records=[];total=Counter()
            ticks=[];rss_samples=[]
            for s,g,row in examples:
                p=ev.call_endpoint(url,row)
                valid=ev.valid_prediction(p) and not(p.get('check_error') or p.get('_error'))
                decision=p['decision'] if valid else 'error'
                total[g+'/'+decision]+=1
                records.append([s,g,decision,reason(p),p.get('risk_category'),p['latency_ms']])
                ticks.append(p['latency_ms'])
                if len(ticks)%20==1:
                    total_rss=0
                    for child in [server,*server.children(recursive=True)]:
                        try:total_rss+=child.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    rss_samples.append(total_rss)
                if len(ticks)%100==0:print(name,len(ticks),flush=True)
            result=dict(total=total,rows=records,p95_ms=float(np.percentile(ticks,95)),
                        max_ms=float(max(ticks)),startup_s=startup,max_sampled_rss_mib=max(rss_samples)/2**20)
            with path.open('x',encoding='utf8') as f:json.dump(result,f,indent=1)
            print(tag,name,total,round(result['p95_ms']),round(result['max_ms']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('tag')
    p.add_argument('suites',nargs='+',choices=['development','historic','holdout','public'])
    args=p.parse_args()
    evaluate(args.tag,args.suites)
