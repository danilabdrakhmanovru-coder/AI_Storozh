"""Train a full-document correction and compare the quantized graph with archived float features."""
import json
import os
import sys
from functools import lru_cache
from zipfile import ZipFile
import joblib
import numpy as np
import psutil
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from app.recipient_model import WEIGHTS,encode,runtime
from app.recipient_fast import features
from .data import RAW,OUT,sha,read
from .train import rows
from app.intent_review import Detector,WEIGHTS as TOKENIZER_DIR
from tokenizers import Tokenizer


BOUNDED=dict(window=258,stride=210,max_windows=3)

@lru_cache(maxsize=1)
def research_runtime():
    directory=RAW/'v27-quantized'
    parts=[]
    for name in read(directory/'int8.json')['parts']:
        with ZipFile(directory/name) as z:parts.append(z.read('chunk'))
    detector=Detector(b''.join(parts),TOKENIZER_DIR/'tokenizer.json')
    tok=Tokenizer.from_file(str(TOKENIZER_DIR/'tokenizer.json'))
    tok.no_truncation()
    return detector.session,tok

def compute(name,texts,geometry=None):
    path=RAW/f'v27-full-{name}.npz'
    import hashlib
    signature=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode('utf8')).hexdigest()
    seal=signature+sha(RAW/'v27-quantized/int8.json')+(json.dumps(geometry,sort_keys=True) if geometry else '')
    if path.exists():
        cached=np.load(path)
        assert str(cached['seal'])==seal
        return cached['x'],cached['covered']
    values=[];covered=[]
    for _,text,_ in texts:
        x,c=encode(*research_runtime(),text,**(geometry or {}))
        values.append(x);covered.append(c)
        if len(values)%100==0:print(name,len(values),flush=True)
    x=np.asarray(values);c=np.array(covered)
    np.savez_compressed(path,x=x,covered=c,seal=seal)
    return x,c


def main(bounded=False):
    rank,cal,dev=rows()
    n=len(read(RAW/'v25-fit.json'))
    a=np.load(RAW/'v25-fit.npz')['x']
    extra,_=compute('rank',rank[n:])
    train=np.concatenate([a,extra])
    y=np.asarray([r[2] for r in rank])
    semantic=make_pipeline(StandardScaler(),LogisticRegression(C=.02,max_iter=1500,class_weight='balanced'))
    semantic.fit(train[:,3:],y)
    native=np.load(RAW/'v25-dev.npz')['x']
    oldextra=np.load(RAW/'v25b-extra.npz')['x']
    manual=cal[len(native)+len(oldextra):]
    tag='bounded-' if bounded else ''
    geometry=BOUNDED if bounded else {}
    cx,_=compute(tag+'cal',manual,geometry)
    x=np.concatenate([native,oldextra,cx])
    cx=features([r[1] for r in cal],x,semantic)
    dx,dc=compute(tag+'dev',dev,geometry)
    vx=features([r[1] for r in dev],dx,semantic)
    cy=np.array([r[2] for r in cal]);vy=np.array([r[2] for r in dev],bool)
    # Check numerical fidelity on development examples that have archived float predictions.
    selected=np.random.default_rng(273).choice(len(native),48,replace=False)
    quant,_=compute('equivalence'+('-bounded-training' if bounded else ''),[cal[i] for i in selected])
    float_scores=native[selected,0];quant_scores=quant[:,0]
    diff=dict(n=len(selected),agreement=int(np.sum((float_scores>=0)==(quant_scores>=0))),
              mean_abs_logit=float(np.mean(abs(float_scores-quant_scores))))
    trials=[]
    best=None
    for label,head in [('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1500))),
                       ('trees',HistGradientBoostingClassifier(max_leaf_nodes=7,l2_regularization=4,max_iter=80,random_state=27))]:
        head.fit(cx,cy)
        p=head.predict_proba(vx)[:,1]
        for hi in (.25,.35,.45,.55,.65,.75):
            pred=p>=hi
            rr=[]
            for source in sorted(set(r[0] for r in dev)):
                mask=np.array([r[0]==source for r in dev])
                rr.append([float(pred[mask&vy].mean()),float(pred[mask&~vy].mean())])
            loss=np.mean([8*max(0,.94-r)+4*max(0,f-.065)+(1-r)+f for r,f in rr])
            trials.append([label,hi,rr])
            if best is None or loss<best[0]:best=loss,label,head,hi
    _,label,head,hi=best
    lo=hi-.06
    cap=float(np.quantile(cx[cy==0,6],.99))
    cfg=dict(head=head,semantic=semantic,upper=hi,lower=lo,max_reference_span=cap,geometry=geometry)
    dest=OUT/'bounded-candidate.joblib' if bounded else RAW/'v27-quantized/full-calibration.joblib'
    joblib.dump(cfg,dest,compress=3)
    result=dict(choice=label,upper=hi,lower=lo,max_reference_span=cap,geometry=geometry,trials=trials,
                calibration_rows=len(cal),fit_rows=len(rank),dev_rows=len(dev),quantization=diff,
                weight_sha=sha(dest))
    (OUT/('bounded-training.json' if bounded else 'full-training.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
    print('full fit',label,hi,diff,flush=True)

if __name__=='__main__':
    p=psutil.Process()
    stage=sys.argv[1] if len(sys.argv)>1 else 'full'
    p.cpu_affinity(sorted(p.cpu_affinity())[-8:] if stage=='cache-bounded' else sorted(p.cpu_affinity())[:4])
    os.environ.update(STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='13')
    if stage=='cache-bounded':
        _,cal,dev=rows()
        n=len(np.load(RAW/'v25-dev.npz')['x'])+len(np.load(RAW/'v25b-extra.npz')['x'])
        compute('bounded-cal',cal[n:],BOUNDED)
        compute('bounded-dev',dev,BOUNDED)
    else:
        main(stage=='bounded')
