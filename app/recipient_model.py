"""Estimate whether an attachment redirects the current assistant's answer. Offline CPU only."""
from functools import lru_cache
from pathlib import Path
import re
import joblib
import numpy as np
from tokenizers import Tokenizer

WEIGHTS=Path(__file__).with_name('weights')/'document-recipient'
MAX_CHARS=12000
WINDOW=384
STRIDE=320
MAX_WINDOWS=4

# Only an explicitly read-only task may clear a previous statistical block.
# Longer or ambiguous user instructions keep all incumbent checks.
READ_ONLY=re.compile(
    r'(?:please\s+)?(?:summari[sz]e|describe|outline|review)\s+(?:(?:this|the|my|attached|enclosed)\s+){0,3}'
    r'(?:document|file|attachment|report|memo|note|record|article|text)|'
    r'(?:пожалуйста[, ]+)?(?:кратко\s+)?(?:перескажи(?:те)?|опиши(?:те)?|резюмируй(?:те)?)\s+'
    r'(?:(?:этот|это|содержимое|содержание|приложенный|приложенного|прикрепленного|прикреплённого)\s+){0,3}'
    r'(?:документ[а]?|вложени[ея]|файл[а]?|текст[а]?|отч[её]т[а]?)',re.I)

def read_only_task(prompt):
    return bool(READ_ONLY.fullmatch(prompt.strip().rstrip('.!?').strip()))

@lru_cache(maxsize=1)
def runtime():
    from .intent_review import WEIGHTS as ENCODER
    from .confidence_review import reviewer
    tok=Tokenizer.from_file(str(ENCODER/'tokenizer.json'))
    tok.no_truncation()
    return reviewer().previous.features.detector.session,tok

def encode(session,tokenizer,text,*,window=WINDOW,stride=STRIDE,max_windows=MAX_WINDOWS,with_review=False):
    tokens=tokenizer.encode(text).ids
    body=tokens[1:-1]
    width=window-2
    last=max(0,len(body)-width)
    starts=list(range(0,last+1,stride))
    if starts[-1]!=last: starts.append(last)
    complete=len(starts)<=max_windows
    if not complete:
        starts=[starts[i] for i in np.linspace(0,len(starts)-1,max_windows).round().astype(int)]
    parts=[[tokens[0],*body[i:i+width],tokens[-1]] for i in starts]
    shape=(len(parts),max(map(len,parts)))
    ids=np.full(shape,tokenizer.token_to_id('<pad>') or 0,dtype=np.int64)
    mask=np.zeros(shape,dtype=np.int64)
    for i,item in enumerate(parts):
        ids[i,:len(item)]=item
        mask[i,:len(item)]=1
    logits,emb=session.run(None,{'input_ids':ids,'attention_mask':mask})
    scores=logits[:,1]-logits[:,0]
    best=int(scores.argmax())
    pooled=np.r_[scores[best],scores.mean(),scores.min(),emb[best],emb.mean(axis=0)]
    if with_review:
        review=np.r_[pooled[:3],scores.std(),np.log1p(len(body)),pooled[3:]]
        return pooled,complete,review
    return pooled,complete

def signals(heads,features,texts):
    model=heads['recipient']
    recipient=model.decision_function(texts)
    active=list(model.classes_).index('agent')
    competing=np.delete(recipient,active,axis=1).max(axis=1)
    semantic=(heads['content'].decision_function(features[:,3:]) if 'content' in heads
              else np.zeros(len(features)))
    return np.column_stack((features[:,0],semantic,recipient[:,active]-competing,
                            heads['contrast'].decision_function(texts)))

def margin(signals_,config):
    return signals_ @ np.asarray(config['weights'])

@lru_cache(maxsize=1)
def trained():
    return joblib.load(WEIGHTS/'model.joblib')

def classify_document(text):
    cfg=trained()
    features,complete=encode(*runtime(),text)
    score=float(margin(signals(cfg,features[None,:],[text]),cfg)[0])
    if score>=cfg['upper']:return 'agent'
    if not complete:return 'uncovered'
    if score<=cfg['lower']:return 'reference'
    return 'uncertain'

def review_attachment(req,verdict,policy,from_model,recheck,predict=None,*,rule='document_recipient_v25',review_vectors=None):
    from .document_review import review as previous
    from .contest import result
    decision=verdict['decision']
    # Explicit provenance comes from the model call site. A matching rule name
    # in a user-supplied policy can never grant a model permission to clear it.
    if decision=='block' and not from_model:return verdict
    if decision=='confirm':return verdict
    docs=req.context.attachments
    if (len(docs)!=1 or req.context.history or not read_only_task(req.user_prompt)
            or 'indirect_injection' not in policy['always_block']
            or not docs[0].content.strip() or len(docs[0].content)>MAX_CHARS):
        return previous(req,verdict,policy,recheck)
    target=(predict or classify_document)(docs[0].content)
    if target=='agent':
        return dict(**result('block','indirect_injection',
             'Модель определила, что вложение задаёт ответ текущему ассистенту.'),
             evidence=[dict(source='attachment',index=0,rule=rule)])
    if target=='reference':
        return recheck() if decision=='block' else verdict
    if target=='uncovered' and decision=='block':return verdict
    if review_vectors is not None:
        return previous(req,verdict,policy,recheck,review_vectors=review_vectors)
    return previous(req,verdict,policy,recheck)

