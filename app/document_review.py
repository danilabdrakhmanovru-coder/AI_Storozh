"""Read-only judgement of embedded instructions in an untrusted text, with a bounded number of windows."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

DIR=Path(__file__).with_name('weights')/'source-documents'
MAX_CHARS=12000
SOFT_DOCUMENT_RULES=frozenset({'document_intent_v20','injection_intent_v20','attachment_passage','attachment_passage_supported',
    'attachment_passage_consensus','document_encoder','learned_bag','isolated_consensus'})


def features(detector,text):
    tokens=detector.tokenizer.encode(text).ids
    body=tokens[1:-1]
    width=146
    starts=list(range(0,max(1,len(body)-width+1),112))
    last=max(0,len(body)-width)
    if starts[-1]!=last:starts.append(last)
    if len(starts)>4:starts=[starts[i] for i in np.linspace(0,len(starts)-1,4).round().astype(int)]
    parts=[[tokens[0],*body[s:s+width],tokens[-1]] for s in starts]
    ids=np.zeros((len(parts),max(map(len,parts))),dtype=np.int64)
    mask=np.zeros_like(ids)
    for i,part in enumerate(parts):
        ids[i,:len(part)]=part;mask[i,:len(part)]=1
    logits,pooled=detector.session.run(None,dict(input_ids=ids,attention_mask=mask))
    raw=logits[:,1]-logits[:,0];best=int(raw.argmax())
    return np.concatenate(([raw[best],raw.mean(),raw.min(),raw.std(),np.log1p(len(body))],pooled[best],pooled.mean(axis=0)))


@lru_cache(maxsize=1)
def model():
    return joblib.load(DIR/'model.joblib')


def probability(text):
    from .confidence_review import reviewer
    config=model()
    x=features(reviewer().previous.features.detector,text)
    return float(config['classifier'].predict_proba(x[None,:])[0,1])


def review(req,verdict,policy,recheck,*,review_vectors=None):
    from .contest import result
    if req.context.history or len(req.context.attachments)>1:
        return verdict
    docs=req.context.attachments
    source='attachment' if docs else 'prompt'
    cat='indirect_injection' if docs else 'prompt_injection'
    text=docs[0].content if docs else req.user_prompt
    if cat not in policy['always_block'] or len(text)>MAX_CHARS or not text.strip():return verdict
    # A direct request already traversed the v23 intent classifier. Examine longer
    # text only when it contains multiple potential passages and was otherwise allowed.
    decision=verdict['decision']
    if not docs and (decision!='allow' or len(text)<650 or verdict.get('_operation_level')):return verdict
    evidence=verdict.get('evidence') or []
    soft=(decision=='block' and docs and verdict['risk_category']==cat and evidence and
          all(v.get('source')=='attachment' and v.get('rule') in SOFT_DOCUMENT_RULES for v in evidence))
    if decision!='allow' and not soft:return verdict
    cfg=model()
    vector=review_vectors.get(text) if review_vectors is not None else None
    value=probability(text) if vector is None else float(cfg['classifier'].predict_proba(vector[None,:])[0,1])
    if decision=='allow' and value>=cfg['block' if docs else 'direct']:
        return dict(**result('block',cat,'Модель нашла инструкцию, меняющую задачу агента внутри предоставленного текста.'),
                    evidence=[dict(source=source,index=0,rule='document_review_v24')])
    if soft and value<=cfg['clear']:
        # Earlier hard rules already inspected the whole document. A separate pass
        # preserves every restriction on the original user instruction and role.
        return recheck()
    return verdict
