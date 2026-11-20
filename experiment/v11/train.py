"""Train a contrastive three-class head on development data plus public ordinary text."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from app.contest import ContestRequest
from app.request_intent import representation
from experiment.v6.train import training_rows
from experiment.v9.train import base_rows
from experiment.v11.data import OLD, OUT, ROLE_CASES, make_row, normal, source, split, evaluation


def training_data():
    external = source()
    safe, _, _ = split(external)
    rows = [*base_rows(), *training_rows(OLD / "training.txt"), *training_rows(OUT / "training.txt")]
    for group, indices in [("ordinary", safe), ("jailbreak", map(int, ROLE_CASES.split()))]:
        for i in indices:
            for field in ("text", "translation"):
                rows.append(make_row(f"train-{i}", group, external[i][field].strip()))
    return rows


def fit(rows, encode, destination, withheld):
    examples = {}
    for row in rows:
        if normal(row["user_prompt"]) in withheld:
            raise ValueError("Evaluation text appeared in the training corpus")
        label = row["risk_category"]
        if label != "jailbreak":
            label = "ordinary" if row["expected_decision"] == "allow" else "other"
        key = encode(ContestRequest.model_validate(row))
        if key in examples and examples[key] != label:
            # The same text can be allowed or forbidden depending on role;
            # this text-only head must defer those cases to policy.
            label = "other"
        examples[key] = label
    features = FeatureUnion([
        ("word", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("intent", TfidfVectorizer(token_pattern=r"(?u)\b(?:jbfeature|jb_request)_\w+\b",
                                  use_idf=False, binary=True)),
    ], transformer_weights={"intent": 0.45})
    model = make_pipeline(features, LinearSVC(class_weight="balanced", random_state=42))
    model.fit(list(examples), list(examples.values()))
    joblib.dump(model, destination, compress=3)
    from collections import Counter
    print(dict(Counter(examples.values())))


if __name__ == "__main__":
    fit(training_data(), representation, OUT / "intent.joblib",
        {normal(r["user_prompt"]) for r in evaluation()})
