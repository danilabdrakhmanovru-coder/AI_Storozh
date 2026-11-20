"""Learn from reviewed v12 mistakes; reserve all v13 evaluation inputs."""
import json
from experiment.v6.train import training_rows
from experiment.v11.data import partition, source, make_row
from experiment.v11.train import fit
from experiment.v12.data import evaluation as diagnostic
from experiment.v12.train import development_rows
from experiment.v13.data import OUT, evaluation, source_texts
from app.request_intent import scoped_features


def development_data():
    rows = development_rows() + training_rows(OUT / "training.txt")
    corpus = source()
    safe, _ = partition(corpus)
    groups = {"ordinary": safe[1750:2350]}
    for line in (OUT.parent / "v11/labels.txt").read_text().splitlines():
        group, indices = line.split("|")
        groups[group] = list(map(int, indices.split()))
    for group, indices in groups.items():
        for index in indices:
            for field in ("text", "translation"):
                rows.append(make_row(f"train-{index}", group, corpus[index][field].strip()))
    # Feedback is limited to the 16 adjudicated v12 cases. Include the
    # paired language from the same external source under its existing label.
    old_predictions = json.loads((OUT.parent / "v12/v12-new.json").read_bytes())["rows"]
    feedback = []
    for row, old in zip(diagnostic(), old_predictions, strict=True):
        assert row["id"] == old["id"]
        if row["risk_category"] not in {"none", "jailbreak"} or old["risk_category"] == row["risk_category"]:
            continue
        feedback.append(row)
        if row["id"].startswith("external-"):
            original = corpus[int(row["id"].split("-")[1])]
            for field in ("text", "translation"):
                rows.append({**row, "user_prompt": original[field].strip()})
    assert len(feedback) == 16
    rows.extend(feedback)
    return rows


if __name__ == "__main__":
    rows = development_data()
    withheld = source_texts(evaluation()) | source_texts(evaluation("final"))
    if source_texts(rows) & withheld:
        raise ValueError("A held-out prompt occurs in training context")
    fit(rows, scoped_features, OUT / "intent.joblib", withheld)
