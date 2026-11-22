"""Classify who is being represented and whether the output is explicitly for review."""
from functools import lru_cache
from pathlib import Path

import joblib

WEIGHTS=Path(__file__).with_name('weights')/'correspondence.joblib'

@lru_cache(maxsize=1)
def model():
    return joblib.load(WEIGHTS)

def clearly_unrelated(names,scores,config):
    other=names.index('other')
    return scores[other]-max(v for i,v in enumerate(scores) if i!=other)>=config['fast_other']

def classify_scores(names,lexical,semantic,config):
    if clearly_unrelated(names,lexical,config):
        return 'other'
    weight=config['semantic_weight']
    values=dict(zip(names,(1-weight)*lexical+weight*semantic))
    label=max(values,key=values.get)
    if label=='other':
        # A clear lexical attribution cue can disagree with general-purpose
        # embeddings, especially when the signer is an unfamiliar proper name.
        c,o=names.index('confirm'),names.index('other')
        if lexical[c]==max(lexical) and lexical[c]-lexical[o]>=.35 and values['other']-values['confirm']<.25:
            return 'confirm'
        return label
    if label=='draft':
        # A close split between review-only and unverified representation needs a human.
        return 'confirm' if values['confirm']>values['draft']-.08 else 'draft'
    if values[label]<config['min_attribution']:
        return 'other' if values['other']>=values['draft'] else 'confirm'
    if label=='fraud' and values[label]-max(values['confirm'],values['draft'],values['other'])<config['fraud_margin']:
        return 'confirm'
    return label

def assess(text):
    cfg=model()
    names=cfg['classes']
    left=cfg['lexical'].decision_function([text])[0]
    if clearly_unrelated(names,left,cfg['config']):
        return 'other'
    from .confidence_review import reviewer
    embedder=reviewer().previous.features.previous.embedder
    right=cfg['semantic'].decision_function(embedder.encode([text]))[0]
    return classify_scores(names,left,right,cfg['config'])

def review(req,verdict,policy):
    from .recipient_model import read_only_task
    from .contest import result
    if verdict['decision']=='block' or read_only_task(req.user_prompt):
        return verdict
    label=assess(req.user_prompt)
    if label=='other':
        return verdict
    if label=='fraud' and 'fraud_social_engineering' in policy['always_block']:
        out=result('block','fraud_social_engineering','Запрошено ложное авторство сообщения или приписывание чужого согласия.')
    elif verdict['decision']=='confirm':
        return verdict
    elif label=='draft':
        if verdict['risk_category']!='none':
            return verdict
        out={**verdict,'explanation':'Запрошен предварительный текст для проверки будущим автором.'}
    else:
        out=result('confirm','sensitive_action','Сообщение составляется от чужого имени. Требуется подтверждение полномочий представителя.')
        out['_operation_level']='medium'
    out['evidence']=[*(verdict.get('evidence') or []),dict(source='prompt',index=0,rule='correspondence_intent')]
    return out
