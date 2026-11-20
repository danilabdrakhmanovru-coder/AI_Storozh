"""CPU inference for the supervised contest category model."""
from functools import lru_cache
from pathlib import Path
import unicodedata

import joblib
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / "experiment/v3/categories.joblib"


def packet_text(req):
    parts = [req.user_prompt]
    parts.extend("history " + m.content for m in req.context.history if m.role in {"user", "assistant"})
    parts.extend("attachment " + a.content for a in req.context.attachments)
    return unicodedata.normalize("NFKC", "\n".join(parts)).casefold()


@lru_cache(maxsize=1)
def load():
    return joblib.load(MODEL)


def predict(req):
    artifact = load()
    model = artifact['model']
    scores = model.decision_function([packet_text(req)])[0]
    order = np.argsort(scores)
    if scores[order[-1]] - scores[order[-2]] < artifact['min_margin']:
        return None
    return model.classes_[order[-1]]
