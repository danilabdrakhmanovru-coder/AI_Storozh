"""CPU document recipient model: score every span, then read at most two spans with ONNX."""
from functools import lru_cache
import joblib
import numpy as np
from pathlib import Path
from app.recipient_model import WEIGHTS, runtime

WIDTH=96
STRIDE=72

@lru_cache(maxsize=1)
def ranker():
    return joblib.load(WEIGHTS/'span-ranker.joblib')

@lru_cache(maxsize=1)
def calibration():
    return joblib.load(Path(__file__).with_name('short-calibration.joblib'))


def features(text,rank=None):
    session,tokenizer=runtime()
    encoded=tokenizer.encode(text)
    ids=encoded.ids
    offsets=encoded.offsets[1:-1]
    last=max(0,len(offsets)-WIDTH)
    begins=list(range(0,last+1,STRIDE))
    if begins[-1]!=last:
        begins.append(last)
    snippets=[text[offsets[i][0]:offsets[min(i+WIDTH,len(offsets))-1][1]] for i in begins] if offsets else ['']
    # Every span and the whole document get a lexical score. Long inputs do
    # not acquire a "fully read" status from the two neural probes alone.
    margins=(rank if rank is not None else ranker()).decision_function([text,*snippets])
    local=margins[1:]
    order=np.argsort(-local,kind='stable')
    chosen=[int(order[0])]
    if len(order)>1:
        chosen.append(next((int(i) for i in order[1:] if abs(begins[i]-begins[chosen[0]])>=WIDTH),
                           int(order[1])))
    windows=[[ids[0],*ids[1+begins[i]:min(len(ids)-1,1+begins[i]+WIDTH)],ids[-1]] for i in chosen]
    batch=np.full((len(windows),max(map(len,windows))),tokenizer.token_to_id('<pad>') or 0,dtype=np.int64)
    mask=np.zeros_like(batch)
    for i,w in enumerate(windows):
        batch[i,:len(w)]=w
        mask[i,:len(w)]=1
    logits,_=session.run(None,{'input_ids':batch,'attention_mask':mask})
    neural=logits[:,1]-logits[:,0]
    # Include disagreement between a local directive and the surrounding prose.
    return np.array([margins[0],max(local),np.mean(local),min(local),max(neural),
                     min(neural),neural[0],local[chosen[0]]-margins[0]],dtype=float)


def classify(text):
    cfg=calibration()
    x=features(text)
    score=float(cfg['head'].predict_proba(x[None,:])[0,1])
    if score>=cfg['upper']:
        return 'agent'
    if score<=cfg['lower'] and x[1]<=cfg['max_reference_span']:
        return 'reference'
    return 'uncertain'
