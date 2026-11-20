"""Compare labeled packets through the contest guard and keep row-level decisions."""
import json
from pathlib import Path

import yaml

from app.contest import ContestRequest, inspect_request, read_policy
from experiment.three_domains.run import summary
from experiment.v4.train import ROOT, lines

OUT = Path(__file__).resolve().parent


def pairs(file="pairs.txt", domain="ferry"):
    for i, line in enumerate((OUT / file).read_text("utf-8").splitlines(), 1):
        label, prompt = line.split("|", 1)
        category, _, variant = label.partition("@")
        role = variant if variant in {"sre", "developer"} else "developer"
        ctx = dict(target_agent="it_ops_agent", user_role=role, history=[], attachments=[])
        if variant == "history":
            ctx["history"] = [dict(role="user", content=prompt)]
            prompt = "Please continue with the previous request."
        if variant == "doc":
            ctx["attachments"] = [dict(content=prompt)]
            prompt = "Summarize the attached memo."
        decision = {"none": "allow", "sensitive_action": "confirm"}.get(category, "block")
        level = {"none": "none", "sensitive_action": "medium", "system_prompt_extraction": "medium"}.get(category, "high")
        yield dict(id=f"{domain}-{i:02}", user_prompt=prompt, context=ctx, risk_category=category,
                   expected_decision=decision, risk_level=level, is_dangerous=decision == "block")


def main(prefix):
    policy = read_policy()
    thresholds = yaml.safe_load((ROOT / "policy.yaml").read_bytes())["thresholds"]
    public = ROOT / "artifacts/astra_public"
    report = {}
    datasets = [("ferry", list(pairs())), ("public", lines(public / "v3-holdout.jsonl")),
                ("v5", lines(public / "v5-holdout.jsonl")), ("v6", lines(ROOT / "experiment/v6/challenge.jsonl"))]
    if prefix != "before":
        datasets.append(("radio", list(pairs("radio.txt", "radio"))))
    for name, rows in datasets:
        preds = []
        for row in rows:
            row.setdefault("is_dangerous", row["expected_decision"] == "block")
            verdict = inspect_request(ContestRequest.model_validate(row), thresholds, policy)
            preds.append(dict(id=row["id"], **verdict))
        report[name] = summary(rows, preds)
        if name in {"ferry", "radio"}:
            (OUT / f"{prefix}-{name}.json").write_text(json.dumps(preds, ensure_ascii=False, indent=2), "utf8")
    (OUT / f"{prefix}-holdouts.json").write_text(json.dumps(report, indent=2), "utf8")
    print(report)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
