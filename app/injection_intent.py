"""CPU text expert for direct and document-borne instruction conflicts (B mode 7)."""
import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
from .attachment_passage import passages

WEIGHTS = Path(__file__).with_name('weights') / 'injection-intent'

@lru_cache(maxsize=1)
def load():
    return joblib.load(WEIGHTS/'model.joblib'), json.loads((WEIGHTS/'thresholds.json').read_bytes())

@lru_cache(maxsize=1)
def auxiliary():
    return joblib.load(WEIGHTS/'auxiliary.joblib')

def opening(text):
    return ' '.join(text.lower().split()[:5])

def focus(row):
    return row[0].casefold()

def neighborhood(row):
    return (row[1][-120:]+' [line] '+row[2][:90]).casefold()

def document_margins(classifier,rows):
    from .attachment_passage import load as passage_model
    import numpy as np
    margins=classifier.decision_function(rows)
    attribution=passage_model().attribution.decision_function(rows)
    return margins - 1.4*np.maximum(0.,attribution)

def document_vote(req, original, classifier, threshold):
    # Always include the old ranker's excerpts. Sample the whole document for additional coverage.
    # Source offsets for excerpts cannot be translated without the original offset map.
    samples={}
    for i,(a,b) in enumerate(zip(original.context.attachments,req.context.attachments)):
        spans=list(passages(a.content))
        step=max(1,(len(spans)+47)//48)
        for s in [*spans[::step],*spans[-1:]]:
            samples.setdefault(s.features,dict(source='attachment',index=i,start=s.start,end=s.end,rule='document_intent_v20'))
        if a.content!=b.content:
            for s in passages(b.content):
                samples.setdefault(s.features,dict(source='attachment',index=i,rule='document_intent_v20'))
    if not samples:return None
    values=document_margins(classifier,list(samples))
    i=int(values.argmax())
    return list(samples.values())[i] if values[i]>=threshold else None

def options(req):
    yield 'prompt',0,req.user_prompt,req.user_prompt
    # The parent cascade has already scanned the original. For large files it passes
    # its contextual excerpts here. Offsets in those projections are not raw offsets.
    for i,attachment in enumerate(req.context.attachments):
        for span in passages(attachment.content):
            focus,before,after=span.features
            yield 'attachment',i,focus,' '.join((before[-100:],focus,after[:80])).strip()

def detect(req,blocked,original):
    model,cut=load()
    document,general=auxiliary()
    if 'indirect_injection' in blocked and req.context.attachments:
        evidence=document_vote(req,original,document,cut['doc_margin'])
        if evidence:
            return ('indirect_injection',evidence),False
    bank=list(dict.fromkeys(options(req)))
    texts=list(dict.fromkeys(text for _,_,a,b in bank for text in (a,b)))
    margins=dict(zip(texts,model.decision_function(texts)))
    labels=model.classes_
    best=None
    neutral=False
    for source,index,a,b in bank:
        scores=margins[a] if source=='prompt' else .65*margins[a]+.35*margins[b]
        pos=int(scores.argmax())
        category=labels[pos]
        if category=='none':
            if source=='prompt':
                # Explanatory, single-question speech only. Compound instructions retain the old checks.
                neutral=bool(scores[pos]>=cut.get('discussion',1.) and re.match(
                    r'^(?:почему|как|зачем|что такое|кто|чем|why|how|what is|who|which)\b',a.strip(),re.I)
                    and not re.search(r'[;\n.!]|\?.+|[,—]\s*(?:а |затем|then |and )',a.strip(),re.I))
            continue
        if source=='prompt' and category=='document_instruction':
            if scores[pos]<cut.get('redirect',.12):continue
            category='prompt_injection'
        if source!='prompt':category='indirect_injection'
        if category not in blocked:continue
        threshold=cut['direct'] if source=='prompt' else cut['attachment']
        strength=float(scores[pos])-threshold
        if strength>=0 and (best is None or strength>best[0]):
            best=(strength,category,dict(source='prompt' if source=='prompt' else 'attachment',index=index,rule='injection_intent_v20'))
    if best is None and not neutral and 'prompt_injection' in blocked:
        s=margins[req.user_prompt]
        rank=sorted((float(score),label) for score,label in zip(s,labels) if label not in ('none','document_instruction') and label in blocked)
        # External labels supply only a binary vote. A specific subtype needs an independent local signal.
        if rank and rank[-1][0]>=cut['subtype_floor'] and general.decision_function([req.user_prompt])[0]>=cut['external_margin']:
            best=(0.,rank[-1][1],dict(source='prompt',index=0,rule='injection_consensus_v20'))
    return (best[1:] if best else None),neutral

