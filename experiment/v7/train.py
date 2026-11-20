"""Train on new service-desk requests in direct and conversational form."""
from pathlib import Path

from experiment.v6.train import main as fit, training_rows

OUT = Path(__file__).resolve().parent


def main(additional=(), destination=OUT / "categories.joblib"):
    rows = training_rows(OUT / "training.txt")
    followups = []
    services = [("rail-hub", "диспетчерская"), ("water-meter", "водоканал"),
                ("ticket-office", "касса"), ("clinic-desk", "регистратура")]
    for i, row in enumerate(rows):
        service, team = services[i % len(services)]
        if row["context"]["attachments"]:
            continue
        prompt = row["user_prompt"]
        # Ticket subjects and host names occur in both benign and malicious
        # traffic. Expose the model to these details on both sides.
        prefixes = (f"Regarding {service}: ", f"For our {service} team, ") if prompt.isascii() else (
            f"По проекту {service}: ", f"У нас в отделе «{team}» вопрос. ")
        for prefix in prefixes:
            followups.append({**row, "user_prompt": prefix + prompt})
        followups.append({
            **row,
            "context": {**row["context"], "history": [{"role": "user", "content": prompt}]},
            "user_prompt": "Please carry on." if prompt.isascii() else "Вернёмся к моему вопросу.",
        })
    fit([*rows, *followups, *additional], destination)


if __name__ == "__main__":
    main()
