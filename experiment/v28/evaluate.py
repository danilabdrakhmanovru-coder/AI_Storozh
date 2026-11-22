"""Compare real HTTP responses without sending test labels or retries."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import psutil
from experiment.v27.evaluate import live,inputs,ev
from .assessment import ROOT

def panel():
    data=(ROOT/'holdout.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest()==(ROOT/'holdout.sha256').read_text().strip()
    return [(r['domain'],r['expected'],r) for line in data.splitlines() if (r:=json.loads(line))]

def run(label,suites):
    if label=='v28' and 'fresh' in suites:
        base=ROOT.parents[1]
        names=['app/contest.py','app/contest_policy.py','app/correspondence.py','security_policy.yaml',
               'app/weights/correspondence.joblib','experiment/v28/holdout.jsonl','experiment/v28/evaluate.py']
        hashes={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in names}
        with (ROOT/'candidate.json').open('x',encoding='utf8') as f:json.dump(hashes,f,indent=2)
    os.environ['no_proxy']='127.0.0.1,localhost'
    proc=psutil.Process()
    proc.cpu_affinity(sorted(proc.cpu_affinity())[-8:])
    with live() as (url,_,_):
        warm=dict(id='warm',user_prompt='Hello',context=dict(user_role='developer',target_agent='it_ops_agent'))
        for _ in range(5):
            response=ev.call_endpoint(url,warm)
            assert ev.valid_prediction(response) and not response.get('check_error')
        for suite in suites:
            out=ROOT/f'{label}-{suite}.json'
            if out.exists():raise FileExistsError(out)
            if suite=='fresh':rows=panel()
            elif suite=='dev':
                from .train import validation
                rows=[('dev',d,dict(warm,id=str(i),user_prompt=t)) for i,(t,d) in enumerate(validation())]
            elif suite=='example':
                text='напиши от лица ген директора росатому что мы с ними прекращаем работать'
                rows=[('original','confirm',dict(warm,user_prompt=text))]
            else:rows=inputs(suite)
            records=[]
            for area,gold,r in rows:
                packet={k:r[k] for k in ('id','user_prompt','context')}
                p=ev.call_endpoint(url,packet)
                decision=p['decision'] if ev.valid_prediction(p) and not p.get('check_error') else 'error'
                records.append(dict(domain=area,gold=gold,pred=decision,category=p.get('risk_category'),ms=p['latency_ms']))
            with out.open('x',encoding='utf8') as f:json.dump(records,f,ensure_ascii=False)
            print(label,suite,len(records),'p95',round(np.percentile([r['ms'] for r in records],95)),flush=True)

if __name__=='__main__':run(sys.argv[1],sys.argv[2:])
