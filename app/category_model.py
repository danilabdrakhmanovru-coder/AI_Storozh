"""CPU inference for the supervised contest category model."""
from functools import lru_cache
from pathlib import Path
import unicodedata

import joblib
import numpy as np
from .intent_features import view

MODEL = Path(__file__).resolve().parents[1] / "experiment/v3/categories.joblib"


def packet_text(req):
    parts = [req.user_prompt]
    parts.extend("history " + m.content for m in req.context.history if m.role in {"user", "assistant"})
    parts.extend("attachment " + a.content for a in req.context.attachments)
    return unicodedata.normalize("NFKC", "\n".join(parts)).casefold()


@lru_cache(maxsize=1)
def load():
    return joblib.load(MODEL)


@lru_cache(maxsize=1)
def load_current():
    return joblib.load(MODEL.parents[1] / 'v5/categories.joblib')


@lru_cache(maxsize=1)
def load_severity():
    return joblib.load(MODEL.parents[1] / 'v5/severity.joblib')


def risk_level(req, category):
    estimator = load_severity()[category]
    if isinstance(estimator, str):
        return estimator
    return str(estimator.predict([view(req)])[0])


def predict(req, *, check_danger=False):
    artifact = load_current()
    model = artifact['model']
    text = [view(req)]
    scores = model.decision_function(text)[0]
    order = np.argsort(scores)
    if check_danger and artifact['danger'].decision_function(text)[0] >= artifact['danger_threshold']:
        return next(model.classes_[i] for i in reversed(order)
                    if model.classes_[i] not in {'none', 'sensitive_action'})
    if scores[order[-1]] - scores[order[-2]] < artifact['min_margin']:
        return None
    return model.classes_[order[-1]]
