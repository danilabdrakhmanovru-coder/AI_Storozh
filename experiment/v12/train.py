"""Retrain the contrastive intent model with independently sourced development inputs."""
from experiment.v6.train import training_rows
from experiment.v11.train import fit, training_data
from experiment.v11.data import OLD, make_row, normal, partition, source, evaluation as previous
from experiment.v12.data import OUT, evaluation
from app.request_intent import current_features


def development_rows():
    rows = training_data() + training_rows(OUT / "training.txt")
    corpus = source()
    safe, _ = partition(corpus)
    roles = (OLD / "final-indices.txt").read_text().splitlines()[0].split("|")[1]
    for group, indices in [("ordinary", safe[950:1500]), ("jailbreak", map(int, roles.split()))]:
        for index in indices:
            for field in ("text", "translation"):
                rows.append(make_row(f"train-{index}", group, corpus[index][field].strip()))
    return rows


if __name__ == "__main__":
    rows = development_rows()
    withheld = {normal(r["user_prompt"]) for r in previous() + evaluation()}
    fit(rows, current_features, OUT / "intent.joblib", withheld)
