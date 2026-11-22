"""Selectively revise a statistical verdict using the incumbent's own confidence."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from . import category_model
from .intent_features import view
from .intent_review import IntentReviewer
from .request_intent import assessment

DIR=Path(__file__).with_name('weights')/'confidence-review'


def incumbent(requests):
    artifact=category_model.load_current()
    model=artifact['model']
    text=[view(r) for r in requests]
    scores=model.decision_function(text)
    safe=scores[:,model.classes_=='none'][:,0]
    alternatives=scores[:,~np.isin(model.classes_,['none','sensitive_action'])]
    danger=artifact['danger'].decision_function(text)
    speech=[assessment(r) for r in requests]
    lengths=np.array([np.log1p(len(r.user_prompt.split())) for r in requests])
    return np.column_stack((safe,safe-alternatives.max(axis=1),danger,
                            [s['ordinary'] for s in speech],[s['other'] for s in speech],
                            [s['jailbreak'] for s in speech],lengths,scores))


def inputs(x, old):
    # Refit from the expert signals, excluding the v22 probability trained on its fusion rows.
    return np.column_stack((old, x))


class ConfidenceReview:
    def __init__(self):
        self.previous=IntentReviewer()
        self.weights=joblib.load(DIR/'correction.joblib')

    def revise(self, req, verdict, blocked):
        from .contest import result
        prior, categories=self.previous.features.extract([req.user_prompt])
        old=incumbent([req])
        x=inputs(prior,old)
        model=self.weights
        score=model['head'].predict_proba(x)[0,1]
        if verdict['decision']=='block':
            if score>model['lower'] or old[0,1]<model['protect_attack']:
                return None
            choice,category='allow','none'
            detail='Классификаторы распознали допустимую задачу. Правила доступа не сработали.'
        elif verdict['decision']=='allow':
            if score<model['upper'] or old[0,1]>model['protect_safe']:
                return None
            local=self.previous.model['local_classes']
            margins=prior[0,2:2+len(local)]
            c=local[int(np.argmax(margins))]
            category=c if c!='none' and max(margins)>=self.previous.model['local_margin'] else categories[0]
            if category not in blocked and prior[0,0]>0:category='prompt_injection'
            if category not in blocked:return None
            choice='block'
            detail=f'Классификаторы выявили запрещённое намерение в запросе. Категория {category}.'
        else:return None
        return dict(**result(choice,category,detail),evidence=[dict(source='prompt',index=0,rule='confidence_review_v23')])


@lru_cache(maxsize=1)
def reviewer():
    return ConfidenceReview()
