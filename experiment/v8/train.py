"""Fit the CPU guard using separately authored examples and neutral conversation history."""
from pathlib import Path
from experiment.v6.train import training_rows
from experiment.v7.train import main as fit

OUT = Path(__file__).resolve().parent


def main():
    rows = training_rows(OUT / "training.txt")
    with_context = []
    for row in rows:
        if row["context"]["attachments"]:
            # A summary prompt also accompanies hostile documents. Vary the
            # wrapper for both labels so the body determines the prediction.
            for wrapper in ("What decisions were recorded?", "Give me a summary of the correspondence.",
                            "Read this email and tell me what the team should follow up on.",
                            "Коротко изложи основные пункты этой переписки."):
                with_context.append({**row, "user_prompt": wrapper})
            continue
        english = row["user_prompt"].isascii()
        history = [{"role": "user", "content": "We are reviewing today's tasks." if english else "Обсудим рабочие вопросы."},
                   {"role": "assistant", "content": "I'm ready." if english else "Готов помочь."}]
        with_context.append({**row, "context": {**row["context"], "history": history}})
    fit([*rows, *with_context], OUT / "categories.joblib")


if __name__ == "__main__":
    main()
