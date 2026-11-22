"""Recipient API and the bounded logit transform used to reproduce the v25b training."""
import numpy as np
from .recipient_fast import classify

def combine(values,settings):
    features=values[:,2:]
    adjustment=features @ np.asarray(settings['coef'])+settings['intercept']
    limit=np.where(adjustment<0,settings['negative_cap'],settings['positive_cap'])
    return values[:,0]+limit*np.tanh(adjustment/limit)

