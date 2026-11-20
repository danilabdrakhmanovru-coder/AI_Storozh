"""CPU inference for the supervised contest category model."""
from functools import lru_cache
from pathlib import Path
import unicodedata

import joblib
import numpy as np
from .intent_features import view
from .jailbreak_intent import detects as detects_jailbreak, may_reclassify
from .request_intent import assessment, role_change

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
    return joblib.load(MODEL.parents[1] / 'v8/categories.joblib')


@lru_cache(maxsize=1)
def load_severity():
    return joblib.load(MODEL.parents[1] / 'v5/severity.joblib')


def risk_level(req, category):
    estimator = load_severity()[category]
    if isinstance(estimator, str):
        return estimator
    return str(estimator.predict([view(req)])[0])


def predict(req, *, check_danger=False, prioritize_jailbreak=True):
    artifact = load_current()
    model = artifact['model']
    text = [view(req)]
    scores = model.decision_function(text)[0]
    order = np.argsort(scores)
    danger = artifact['danger'].decision_function(text)[0] if check_danger else None
    if check_danger and danger >= artifact['danger_threshold']:
        label = next(model.classes_[i] for i in reversed(order)
                     if model.classes_[i] not in {'none', 'sensitive_action'})
    elif scores[order[-1]] - scores[order[-2]] < artifact['min_margin']:
        label = None
    else:
        label = model.classes_[order[-1]]
    if not req.context.attachments and prioritize_jailbreak:
        intent = assessment(req)
        threshold = 0.05 if role_change(req.user_prompt) else 0.15
        if intent['jailbreak'] >= threshold and intent['jailbreak'] - max(intent['other'], intent['ordinary']) >= 0.30:
            return 'jailbreak'
        # Correct low-confidence text-only flags. Role/tool decisions and the
        # independent last risk check still apply in the gateway.
        if (check_danger and danger < 0.30 and not req.context.history
                and intent['ordinary'] >= 0.45 and intent['other'] < -0.35
                and intent['jailbreak'] < -0.35):
            return 'none'
    if label == 'jailbreak':
        return label
    if (label in {None, 'none', 'sensitive_action'} or
            (prioritize_jailbreak and may_reclassify(req))) and detects_jailbreak(req):
        return 'jailbreak'
    return label
