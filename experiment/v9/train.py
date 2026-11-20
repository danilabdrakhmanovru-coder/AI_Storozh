"""Fit an additional jailbreak decision boundary on the existing train split and new pairs."""
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from app.contest import ContestRequest
from app.jailbreak_model import MODEL, model_text
from experiment.v4.train import ROOT, lines
from experiment.v6.train import training_rows

OUT = Path(__file__).resolve().parent


def main(*, extra_rows=(), encode=model_text, destination=MODEL):
    excluded = set(json.loads((ROOT / "experiment/v3/split.json").read_bytes())["holdout_ids"])
    rows = [r for r in lines(ROOT / "artifacts/astra_public/dev.jsonl") if r["id"] not in excluded]
    rows.extend(r for r in lines(ROOT / "experiment/synthetic1000/cases.jsonl") if r["tags"][0] == "bank")
    for version in ("v4", "v6", "v7", "v8", "v9"):
        rows.extend(training_rows(OUT.parent / version / "training.txt"))
    new_rows = training_rows(OUT / "training.txt")
    for r in new_rows:
        english = r["user_prompt"].isascii()
        context = {**r["context"], "history": [
            {"role": "user", "content": "Can we start the session?" if english else "Начнём обсуждение?"},
            {"role": "assistant", "content": "Yes." if english else "Да."}]}
        rows.append({**r, "context": context})

    rows.extend(extra_rows)
    examples = {}
    for r in rows:
        text = encode(ContestRequest.model_validate(r))
        label = r["risk_category"] == "jailbreak"
        if text in examples and examples[text] != label:
            raise ValueError("Conflicting jailbreak labels for identical input")
        examples[text] = label

    model = make_pipeline(FeatureUnion([
        ("word", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ]), LinearSVC(class_weight="balanced", random_state=42))
    model.fit(list(examples), list(examples.values()))
    joblib.dump(model, destination, compress=3)
    print(f"Unique train inputs: {len(examples)} including {sum(examples.values())} jailbreaks")


if __name__ == "__main__":
    main()
