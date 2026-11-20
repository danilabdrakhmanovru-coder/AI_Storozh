"""Run from the project root: python -m experiment.compare"""
import json
import time
from collections import Counter
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from app.model import Ollama

ROOT = Path(__file__).resolve().parent

def load(name):
    lines = (ROOT / name).read_text(encoding="utf-8-sig").splitlines()
    return [line.split("|", 1) for line in lines if line.strip()]

def decision(score):
    return "block" if score >= .75 else "confirm" if score >= .35 else "allow"

def main():
    train, test = load("train.txt"), load("test.txt")
    assert not ({t.casefold() for _, t in train} & {t.casefold() for _, t in test})
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
                          LogisticRegression(C=20, random_state=0))
    model.fit([t for _, t in train], [label for label, _ in train])
    joblib.dump(model, ROOT / "classifier.joblib")
    judge = Ollama()
    judge.url = "http://127.0.0.1:11434"
    judge.model = "qwen2.5:7b"
    judge.enabled = True
    if not judge.status()["available"]:
        raise RuntimeError("Local Qwen is unavailable")
    rows = []
    for label, text in test:
        start = time.perf_counter()
        score = float(model.predict_proba([text])[0, list(model.classes_).index("threat")])
        elapsed = (time.perf_counter()-start)*1000
        verdict = judge.judge({"text":text})
        row = {"text":text, "label":label, "local":decision(score), "local_score":score,
               "local_ms":elapsed, "qwen":decision(verdict["risk"]) if verdict["status"]=="ok" else "error",
               "qwen_details":verdict}
        rows.append(row)
        print(len(rows), label, row["local"], row["qwen"], flush=True)
    result = {"model":judge.model,"train_size":len(train),"thresholds":[.35,.75],"rows":rows}
    result["counts"] = {m:{label:dict(Counter(r[m] for r in rows if r["label"]==label))
                            for label in ("safe","threat")} for m in ("local","qwen")}
    (ROOT/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
