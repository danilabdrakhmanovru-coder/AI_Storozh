"""Paired text-only benchmark. Run python -m experiment.v2.run from the root."""
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.base import clone
from experiment.compare import ROOT, load, decision
from app.model import Ollama

OUT = Path(__file__).resolve().parent

def main():
    old = joblib.load(ROOT / "classifier.joblib")
    train = load("train.txt") + load("v2/additional.txt")
    test = load("v2/test.txt")
    texts = [text.strip().casefold() for _, text in train + test + load("test.txt")]
    assert len(texts) == len(set(texts))
    new = clone(old)
    new.fit([text for _, text in train], [label for label, _ in train])
    joblib.dump(new, OUT / "classifier.joblib")
    judge = Ollama()
    judge.url = "http://127.0.0.1:11434"
    judge.model = "qwen2.5:7b"
    judge.enabled = True
    assert judge.status()["available"], "Local Qwen unavailable"

    rows = []
    for label, text in test:
        row = {"label": label, "text": text}
        for name, classifier in [("v1", old), ("v2", new)]:
            score = classifier.predict_proba([text])[0, list(classifier.classes_).index("threat")]
            row[name] = decision(score)
            row[name + "_score"] = float(score)
        verdict = judge.judge({"text": text})
        row["qwen_details"] = verdict
        row["qwen"] = decision(verdict["risk"]) if verdict["status"] == "ok" else "error"
        rows.append(row)
        print(len(rows), label, row["v1"], row["v2"], row["qwen"], flush=True)
    counts = {name: {label: dict(Counter(r[name] for r in rows if r["label"] == label))
                     for label in ("safe", "threat")} for name in ("v1","v2","qwen")}
    result = {"train": dict(Counter(label for label, _ in train)), "thresholds": [.35,.75],
              "qwen_model": judge.model, "counts": counts, "rows": rows}
    (OUT/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(counts)

if __name__ == "__main__":
    main()
