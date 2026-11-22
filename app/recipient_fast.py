"""Classify the addressee of a complete document with bounded CPU inference."""
from functools import lru_cache
import json
import joblib
import numpy as np
from .recipient_model import WEIGHTS, encode, runtime, trained, signals


@lru_cache(maxsize=1)
def ranker():
    return joblib.load(WEIGHTS/'span-ranker.joblib')


@lru_cache(maxsize=1)
def calibration():
    model=joblib.load(WEIGHTS/'full-calibration.joblib')
    encoder=json.loads((WEIGHTS.parent/'intent-review'/'encoder.json').read_bytes())
    if model.get('encoder_sha256')!=encoder['graph_sha256']:
        raise ValueError('Калибровка адресата и энкодер имеют разные версии')
    return model


@lru_cache(maxsize=1)
def legacy():
    cfg=json.loads((WEIGHTS/'bounded.json').read_bytes())
    return cfg,joblib.load(WEIGHTS/'calibration-projection.joblib')


def lexical(text):
    # Score the whole text and overlapping spans, including the exact suffix.
    width=560
    last=max(len(text)-width,0)
    starts=list(range(0,last+1,400))
    if starts[-1]!=last:starts.append(last)
    scores=ranker().decision_function([text,*[text[i:i+width] for i in starts]])
    return [scores[0],max(scores[1:]),np.mean(scores[1:])]


def features(texts,x,semantic):
    # Keep the original locally calibrated encoder as an independent signal.
    from .recipient_calibration import combine
    cfg,projection=legacy()
    sig=signals(trained(),x,texts)
    context=projection.transform(x[:,3:])[:,:cfg['dimensions']]
    old=combine(np.column_stack((sig,context)),cfg)
    aux=semantic.decision_function(x[:,3:])
    return np.column_stack((x[:,:3],old,aux,np.asarray([lexical(t) for t in texts])))


def decision(config,z,complete):
    score=float(config['head'].predict_proba(z[None,:])[0,1])
    if score>=config['upper']:return 'agent'
    if not complete:return 'uncovered'
    if score<=config['lower'] and z[6]<=config['max_reference_span']:return 'reference'
    return 'uncertain'


def classify(text,*,review_vectors=None):
    config=calibration()
    x,complete,review=encode(*runtime(),text,with_review=True,**config.get('geometry',{}))
    z=features([text],x[None,:],config['semantic'])[0]
    target=decision(config,z,complete)
    if review_vectors is not None and target in ('uncertain','uncovered'):
        review_vectors[text]=review
    return target
