"""Measure the four recipient paths on development only and select decision thresholds there."""
import json
import os
import sys
from unittest.mock import patch
import numpy as np
import joblib
import psutil

from .data import OUT, RAW, ROOT, cases, read, sha
from app.recipient_fast import features
from app.contest import ContestRequest, inspect_request, parse_policy
from .evaluate import CTX


def paths():
    raw=(ROOT/'security_policy.yaml').read_bytes()
    limits=parse_policy(raw)['thresholds']
    results=[]
    for i,(s,t,y) in enumerate(cases('development')):
        req=ContestRequest(user_prompt='Summarize the attached document.',context={**CTX,'attachments':[dict(content=t)]})
        d={}
        for state in ('reference','agent','uncertain','uncovered'):
            with patch('app.recipient_calibration.classify',return_value=state):
                d[state]=inspect_request(req,limits,raw)['decision']
        results.append([s,y,d])
        if (i+1)%50==0:print('paths',i+1,flush=True)
    with (OUT/'dev-paths.json').open('x',encoding='utf8') as f:json.dump(results,f)

def choose():
    cfg=joblib.load(OUT/'float-candidate.joblib')
    saved=np.load(RAW/'v27-original-dev.npz')
    dev=cases('development')
    z=features([r[1] for r in dev],saved['x'],cfg['semantic'])
    prob=cfg['head'].predict_proba(z)[:,1]
    refs=read(OUT/'dev-paths.json')
    source=np.asarray([r[0] for r in dev]);y=np.array([r[2] for r in dev],bool)
    candidates=[(cfg['upper'],cfg['lower'])]
    candidates+= [(h,l) for h in (.4,.55,.7,.85,.93) for l in (.25,.45,.65,.83) if l<h]
    best=None;trials=[]
    for hi,lo in candidates:
        state=np.select([prob>=hi,~saved['covered'],(prob<=lo)&(z[:,6]<=cfg['max_reference_span'])],
                        ['agent','uncovered','reference'],default='uncertain')
        pred=np.array([r[2][m] for r,m in zip(refs,state)])=='block'
        rates=[(float(pred[y&(source==s)].mean()),float(pred[(~y)&(source==s)].mean())) for s in sorted(set(source))]
        recall=pred[y].mean();fpr=pred[~y].mean()
        # Prioritise the release constraints, then worst-source recall and cost.
        cost=(max(0,.90-recall)*10+max(0,fpr-.10)*4,
              sum(2*max(0,.90-r)+max(0,f-.10) for r,f in rates),
              (1-recall)*3+fpr)
        if best is None or cost<best[0]:best=cost,hi,lo
        trials.append([hi,lo,rates,round(float(recall),4),round(float(fpr),4)])
    _,hi,lo=best
    cfg.update(upper=hi,lower=lo)
    dst=OUT/'pipeline-candidate.joblib'
    joblib.dump(cfg,dst,compress=3)
    (OUT/'pipeline-training.json').write_text(json.dumps(dict(upper=hi,lower=lo,trials=trials,
                sha=sha(dst)),indent=2),encoding='utf8')
    print('selected', [t for t in trials if t[:2]==[hi,lo]][0],flush=True)

if __name__=='__main__':
    os.environ.update(STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='13')
    p=psutil.Process();p.cpu_affinity(sorted(p.cpu_affinity())[:4])
    (paths if sys.argv[1]=='paths' else choose)()
