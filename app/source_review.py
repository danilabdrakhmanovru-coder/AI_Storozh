"""Scores from source-specific experts and semantic reference descriptions for B v21."""
from pathlib import Path
import numpy as np
from .injection_intent import load as local_model,auxiliary

def local_scores(texts):
    model,_=local_model()
    all_scores=model.decision_function(texts)
    scores=all_scores[:,model.classes_!='none'].max(axis=1)-all_scores[:,model.classes_=='none'][:,0]
    return np.column_stack((scores,auxiliary()[1].decision_function(texts)))

def columns(embedded,texts,heads,probes=None):
    values=[np.column_stack([h.decision_function(embedded) for h in heads['experts']]),
            heads['lexical'].decision_function(texts)[:,None],local_scores(texts)]
    for anchors in heads['anchors']:
        distances=embedded@anchors.T
        values.append(np.column_stack((distances.max(axis=1),distances.mean(axis=1))))
    if probes is not None:values.append(probes)
    return np.hstack(values)

