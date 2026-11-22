"""Fit a whole-document/span ranker and train a bounded semantic recipient model on disjoint rows."""
from collections import Counter
import json
import os
import random
import sys
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion,make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import psutil

from .data import OUT, RAW, SEED, cases, corpus, historical, read, sha
from .short_model import features,WEIGHTS

def prepare():
    from experiment.v20.transfer_sources import Near
    from experiment.v22.data import add
    from experiment.v26.data import previous_texts
    rng=random.Random(SEED+1)
    idx=Near(set(previous_texts()))
    # The panel indices are used solely for duplicate exclusion.
    for items in (historical(),cases('holdout'),cases('development')):
        for _,t,_ in items:add(idx,t)
    extra=[]
    for source,pool in corpus().items():
        order=list(range(len(pool)));rng.shuffle(order)
        taken=Counter()
        for i in order:
            t,y=pool[i]
            if 20<=len(t)<=12000 and taken[y]<250 and not idx.contains(t):
                add(idx,t)
                extra.append([source,i,y])
                taken[y]+=1
    rng.shuffle(extra)
    groups=Counter()
    rank,cal=[],[]
    for s,i,y in extra:
        groups[s,y]+=1
        (cal if groups[s,y]%3==0 else rank).append([s,i])
    info=dict(rank=rank,cal=cal,panels_sha=sha(OUT/'panels.json'))
    with (OUT/'fit-index.json').open('x',encoding='utf8') as f:json.dump(info,f,indent=1)
    print('Extra fit',len(rank),'calibration',len(cal),flush=True)

def rows():
    c=corpus()
    index=read(OUT/'fit-index.json')
    assert index['panels_sha']==sha(OUT/'panels.json')
    rank=[r[:3] for r in read(RAW/'v25-fit.json')]+[(s,*c[s][i]) for s,i in index['rank']]
    rank.extend(('reference',t,0) for t in (OUT/'human.txt').read_text('utf8').splitlines() if t)
    cal=[r[:3] for r in read(RAW/'v25-dev.json')]+read(RAW/'v25b-calibration.json')+[(s,*c[s][i]) for s,i in index['cal']]
    return rank,cal,cases('development')

def train():
    rank,cal,val=rows()
    vector=FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=45000)),
                         ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=60000))])
    ranker=make_pipeline(vector,LinearSVC(C=1.6,class_weight='balanced',random_state=SEED))
    ranker.fit([r[1] for r in rank],[r[2] for r in rank])
    joblib.dump(ranker,WEIGHTS/'span-ranker.joblib',compress=3)
    for name,group in [('cal',cal),('dev',val)]:
        x=[]
        for _,t,_ in group:
            x.append(features(t,ranker))
            if len(x)%150==0:print(name,len(x),flush=True)
        np.savez_compressed(RAW/f'v27-{name}.npz',x=x,y=[r[2] for r in group])
    fit()

def fit():
    _,cal,val=rows()
    a,b=[np.load(RAW/f'v27-{part}.npz') for part in ('cal','dev')]
    x,y=a['x'],a['y']
    vx,vy=b['x'],b['y'].astype(bool)
    train_groups=Counter((r[0].split('/')[0],r[2]) for r in cal)
    w=np.array([1/np.sqrt(train_groups[r[0].split('/')[0],r[2]]) for r in cal]);w/=w.mean()
    best=None
    trials=[]
    for label,model in [('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000))),
                        ('trees',HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=7,min_samples_leaf=16,l2_regularization=3,random_state=SEED))]:
        if label=='logistic':model.fit(x,y,logisticregression__sample_weight=w)
        else:model.fit(x,y,sample_weight=w)
        p=model.predict_proba(vx)[:,1]
        for hi in (.35,.45,.55,.65,.75,.85):
            pred=p>=hi
            source_rates=[]
            for source in sorted(set(r[0] for r in val)):
                mask=np.array([r[0]==source for r in val])
                source_rates.append([float(pred[mask&vy].mean()),float(pred[mask&~vy].mean())])
            loss=np.mean([6*max(0,.94-r)+4*max(0,f-.07)+(1-r)+f for r,f in source_rates])
            trials.append([label,hi,source_rates])
            if best is None or loss<best[0]:
                best=(loss,label,model,hi)
    _,label,head,hi=best
    lo=min(hi-.06,.4)
    cap=float(np.quantile(x[y==0,1],.98))
    joblib.dump(dict(head=head,upper=hi,lower=lo,max_reference_span=cap),OUT/'short-calibration.joblib',compress=3)
    out=dict(train_rows=len(rows()[0]),cal_rows=len(cal),development_rows=len(val),choice=label,upper=hi,lower=lo,
             max_reference_span=cap,trials=trials,weights={p.name:sha(p) for p in WEIGHTS.glob('span-*.joblib')})
    (OUT/'training.json').write_text(json.dumps(out,indent=1),encoding='utf8')
    print(label,hi,lo,'training complete',flush=True)

if __name__=='__main__':
    proc=psutil.Process()
    proc.cpu_affinity(sorted(proc.cpu_affinity())[:4])
    {'prepare':prepare,'train':train,'fit':fit}[sys.argv[1]]()
