"""Fit the CPU intent head on reviewed v13 feedback; exclude every new evaluation prompt."""
import json

from app.contest import CONTINUATION
from app.request_intent import intent_features
from experiment.v6.train import training_rows
from experiment.v11.data import make_row, partition, source
from experiment.v11.train import fit
from experiment.v13.data import JAILBREAK, evaluation as previous, source_texts
from experiment.v13.train import development_data
from experiment.v14.data import OUT, evaluation


def feedback_rows(corpus):
    predictions = json.loads((OUT.parent / "v13/v13-final.json").read_bytes())["rows"]
    corrected = []
    for row, verdict in zip(previous("final"), predictions, strict=True):
        assert row["id"] == verdict["id"]
        if row["risk_category"] == verdict["risk_category"]:
            continue
        assert row["risk_category"] in {"none", "jailbreak"}
        context = row["context"]
        if CONTINUATION.fullmatch(row["user_prompt"]) and context.get("history"):
            row = {**row, "user_prompt": context["history"][-1]["content"],
                   "context": {**context, "history": context["history"][:-1]}}
        corrected.append(row)
    assert len(corrected) == 15
    rows = list(corrected)
    for row in corrected:
        if row["id"].startswith("external-"):
            source_row = corpus[int(row["id"].split("-")[1])]
            rows.extend({**row, "user_prompt": source_row[f].strip()} for f in ("text", "translation"))
    return rows


if __name__ == "__main__":
    corpus = source()
    rows = development_data() + training_rows(OUT / "training.txt") + feedback_rows(corpus)
    safe, _ = partition(corpus)
    for group, indices in [("ordinary", safe[3000:3900]), ("jailbreak", map(int, JAILBREAK.split()))]:
        for i in indices:
            for field in ("text", "translation"):
                rows.append(make_row(f"train-{i}", group, corpus[i][field].strip()))
    # The first new cohort became development data after the recorded
    # 536-packet trial. Learn from its missed categories and false alarms.
    trial = json.loads((OUT / "initial-final.json").read_bytes())["rows"]
    for row, prediction in zip(evaluation(), trial, strict=True):
        assert row["id"] == prediction["id"]
        if row["risk_category"] in {"none", "jailbreak"} and row["risk_category"] != prediction["risk_category"]:
            ctx = row["context"]
            if CONTINUATION.fullmatch(row["user_prompt"]) and ctx.get("history"):
                row = {**row, "user_prompt": ctx["history"][-1]["content"],
                       "context": {**ctx, "history": ctx["history"][:-1]}}
            rows.append(row)
    withheld = source_texts(evaluation("confirmation"))
    if withheld & source_texts(rows):
        raise ValueError("Held-out content occurs in a training prompt or conversation")
    fit(rows, intent_features, OUT / "intent.joblib", withheld)
