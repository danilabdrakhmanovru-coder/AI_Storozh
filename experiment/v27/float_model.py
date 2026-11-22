"""Fit the release head on the original encoder; quantization equivalence is measured separately."""
import json
import os
import sys
import hashlib
import joblib
import numpy as np
import psutil
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from .train import rows
from .data import OUT,RAW,read,sha
from app.recipient_model import WEIGHTS,encode,runtime
from app.recipient_fast import features

GEOMETRY=dict(window=258,stride=210,max_windows=3)
GRAPH=json.loads((WEIGHTS.parent/'intent-review'/'encoder.json').read_bytes())['graph_sha256']

def sample(rows,n,seed):
    idx=np.random.default_rng(seed).permutation(len(rows))[:min(n,len(rows))]
    return [rows[i] for i in sorted(idx)]

def get(name,rows,geometry):
    dst=RAW/f'v27-original-{name}.npz'
    signature=hashlib.sha256(json.dumps([rows,geometry,GRAPH],ensure_ascii=False).encode()).hexdigest()
    if dst.exists():
        saved=np.load(dst)
        assert str(saved['signature'])==signature
        return saved['x'],saved['covered']
    x=[];covered=[]
    for _,text,_ in rows:
        v,c=encode(*runtime(),text,**geometry)
        x.append(v);covered.append(c)
        if len(x)%60==0:print(name,len(x),flush=True)
    x=np.asarray(x);covered=np.asarray(covered)
    np.savez_compressed(dst,x=x,covered=covered,signature=signature)
    return x,covered

def datasets():
    rank,cal,dev=rows()
    base=len(read(RAW/'v25-fit.json'))
    cbase=len(read(RAW/'v25-dev.json'))+len(read(RAW/'v25b-calibration.json'))
    return rank[:base],sample(rank[base:],180,2722),cal[:cbase],sample(cal[cbase:],160,2723),dev

def fit():
    base,ext,cal,add,dev=datasets()
    x0=np.load(RAW/'v25-fit.npz')['x']
    xe,_=get('fit',ext,GEOMETRY)
    fx=np.concatenate([x0,xe])
    semantic=make_pipeline(StandardScaler(),LogisticRegression(C=.012,max_iter=1000,class_weight='balanced'))
    semantic.fit(fx[:,3:],[r[2] for r in base+ext])
    a=np.load(RAW/'v25-dev.npz')['x'];b=np.load(RAW/'v25b-extra.npz')['x']
    c,_=get('cal',add,GEOMETRY)
    tx=np.concatenate([a,b,c])
    cal=cal+add
    x=features([r[1] for r in cal],tx,semantic)
    y=np.array([r[2] for r in cal])
    v,covered=get('dev',dev,GEOMETRY)
    vx=features([r[1] for r in dev],v,semantic)
    source=np.array([r[0] for r in dev]);vy=np.array([r[2] for r in dev],bool)
    trials=[];best=None
    for name,head in [('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000))),
                      ('trees',HistGradientBoostingClassifier(max_leaf_nodes=7,max_iter=80,l2_regularization=4,random_state=27))]:
        head.fit(x,y)
        prob=head.predict_proba(vx)[:,1]
        for hi in (.25,.35,.45,.55,.65,.75,.85):
            agent=prob>=hi
            rates=[]
            for s in sorted(set(source)):
                mask=source==s
                rates.append([float(agent[mask&vy].mean()),float(agent[mask&~vy].mean())])
            loss=sum(6*max(0,.94-r)+4*max(0,f-.045)+(1-r)+f for r,f in rates)
            trials.append([name,hi,rates])
            if best is None or loss<best[0]:best=(loss,head,name,hi)
    _,head,name,hi=best
    # The lexical guard applies to reference decisions even at low model probability.
    cap=float(np.quantile(x[y==0,6],.995))
    cfg=dict(head=head,semantic=semantic,upper=hi,lower=hi-.05,max_reference_span=cap,
             geometry=GEOMETRY,encoder_sha256=GRAPH)
    dst=OUT/'float-candidate.joblib'
    joblib.dump(cfg,dst,compress=3)
    correction=read(OUT/'bounded-training.json')['quantization']
    result=dict(choice=name,upper=hi,lower=cfg['lower'],max_reference_span=cap,geometry=GEOMETRY,
                fit_rows=len(fx),rank_rows=len(rows()[0]),calibration_rows=len(cal),dev_rows=len(dev),trials=trials,
                quantization=correction,encoder_sha256=GRAPH,weight_sha=sha(dst),
                covered_in_dev=int(covered.sum()))
    (OUT/'float-training.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('float',name,hi,[t for t in trials if t[:2]==[name,hi]],flush=True)

if __name__=='__main__':
    proc=psutil.Process()
    stage=sys.argv[1]
    proc.cpu_affinity(sorted(proc.cpu_affinity())[:4] if stage=='fit-input' else sorted(proc.cpu_affinity())[-8:])
    os.environ.update(STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='13')
    if stage=='fit-input':
        _,ext,_,_,_=datasets()
        get('fit',ext,GEOMETRY)
    elif stage=='dev-input':
        _,_,_,add,dev=datasets()
        get('cal',add,GEOMETRY)
        get('dev',dev,GEOMETRY)
    else:fit()
