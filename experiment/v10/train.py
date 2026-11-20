"""Refit the v9 binary SVM with contrastive role, waiver and legitimate requests."""
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from app.jailbreak_intent import MODEL, features
from experiment.v6.train import training_rows
from experiment.v9.train import main as fit

if __name__ == "__main__":
    rows = training_rows(Path(__file__).with_name("training.txt"))
    turns = []
    for row in rows:
        english = row["user_prompt"].isascii()
        ctx = {**row["context"], "history": [
            {"role": "user", "content": "We have a question for the presenter." if english else "Вопрос к ведущему."},
            {"role": "assistant", "content": "Go ahead." if english else "Слушаю вас."}]}
        turns.append({**row, "context": ctx})
    # Normalize intent features separately, so long contextual descriptions
    # do not drown out the evidence for a requested change of safety behavior.
    intent = TfidfVectorizer(token_pattern=r"(?u)\b(?:jbfeature|jb_request)_\w+\b",
                             use_idf=False, binary=True)
    fit(extra_rows=[*rows, *turns], encode=features, destination=MODEL,
        extra_features=[("intent", intent)], feature_weights={"intent": 0.45})
