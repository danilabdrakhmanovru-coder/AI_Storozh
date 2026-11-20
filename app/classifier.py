"""Inference with the bundled v2 text classifier trained in experiment/v2."""
from functools import lru_cache
from pathlib import Path
import logging

MODEL_PATH = Path(__file__).resolve().parents[1] / "experiment/v2/classifier.joblib"

@lru_cache(maxsize=1)
def load_classifier():
    import joblib
    return joblib.load(MODEL_PATH)

def classify(text):
    try:
        pipeline = load_classifier()
        index = list(pipeline.classes_).index("threat")
        score = float(pipeline.predict_proba([text])[0, index])
        return {"status": "ok", "risk": score, "model": "Классификатор v2"}
    except Exception:
        logging.getLogger(__name__).exception("Classifier inference failed")
        return {"status": "error", "model": "Классификатор v2",
                "reason": "Классификатор не завершил проверку. Требуется повторная проверка."}

def evaluate_stage(result, thresholds):
    if result["status"] != "ok":
        return {**result, "decision": "confirm"}
    score = result["risk"]
    if score >= thresholds["block"]:
        decision = "block"
        reason = f"Оценка {score:.3f} достигла порога блокировки {thresholds['block']:g}."
    elif score >= thresholds["confirm"]:
        decision = "confirm"
        reason = f"Оценка {score:.3f} достигла порога подтверждения {thresholds['confirm']:g}."
    else:
        decision = "allow"
        reason = f"Оценка {score:.3f} ниже порога подтверждения {thresholds['confirm']:g}."
    return {**result, "decision": decision, "reason": result.get("reason", reason)}
