"""Rank the complete attachment, retain source offsets, and give the expensive fallback bounded local evidence."""
from functools import lru_cache
from pathlib import Path
import json
import joblib
import numpy as np
from .attachment_passage import load as passage_head
from .document_encoder import detect as document_detect

WEIGHTS=Path(__file__).with_name('weights')/'attachment-bag'
SHORT=900
def excerpt(text,candidates):
    # Retain source order, the document opening and contextual attribution around the two leading spans.
    ranges=[(0,min(110,len(text)))]
    for *_,s in candidates[:2]:
        ranges.append((max(0,s.start-90),min(len(text),s.end+64)))
    merged=[]
    for a,b in sorted(ranges):
        if merged and a<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    chars=[];offsets=[]
    for a,b in merged:
        if chars:chars.append('\n');offsets.append(a)
        chars.append(text[a:b]);offsets.extend(range(a,b))
    return ''.join(chars),offsets

@lru_cache(maxsize=1)
def model():
    return joblib.load(WEIGHTS/'model.joblib'),json.loads((WEIGHTS/'thresholds.json').read_bytes())

def bag_features(candidates):
    # These are detector margins, not predicted severities, roles or gold labels.
    local=local_svm().decision_function([s.features[0] for *_,s in candidates])
    return np.asarray([[linear,neural,svm,quote,max(0.,quote),linear*max(0.,quote),neural*max(0.,quote)]
                       for (linear,neural,prior,quote,_),svm in zip(candidates,local)],dtype=float)

@lru_cache(maxsize=1)
def local_svm():
    return joblib.load(WEIGHTS.parent/'document-encoder'/'paired-text.joblib')

def score_attachment(attachment, *, learned, isolated=False):
    head=passage_head()
    text=attachment.content
    cand=head.candidates(text)
    hit=head.detect([attachment],[cand])
    if hit:return hit,attachment
    if len(text)<=SHORT and not isolated:
        return document_detect([attachment]),attachment
    if not cand:return None,attachment.model_copy(update={'content':''})
    if learned:
        estimator,config=model()
        values=estimator.decision_function(bag_features(cand))
        i=int(np.argmax(values))
        if values[i]>=config['margin']:
            s=cand[i][-1]
            return dict(source='attachment',start=s.start,end=s.end,rule='learned_bag'),attachment
        if isolated:
            # Require positive document and attribution evidence before dropping neighbors.
            for i in np.argsort(-values)[:2]:
                _,_,_,quote,s=cand[i]
                if values[i]<=0 or quote>0:
                    continue
                single=attachment.model_copy(update={'content':text[s.start:s.end]})
                if document_detect([single]):
                    return dict(source='attachment',start=s.start,end=s.end,rule='isolated_consensus'),attachment
    if len(text)<=SHORT:
        return document_detect([attachment]),attachment
    compact,offsets=excerpt(text,cand)
    doc=attachment.model_copy(update={'content':compact})
    hit=document_detect([doc])
    if hit:
        hit.update(start=offsets[hit['start']],end=offsets[hit['end']-1]+1)
    return hit,doc

def evaluate(attachments, *, learned, isolated=False):
    projected=[]
    for i,a in enumerate(attachments):
        hit,doc=score_attachment(a,learned=learned,isolated=isolated)
        if hit:return {**hit,'index':i},None
        projected.append(doc)
    return None,projected

