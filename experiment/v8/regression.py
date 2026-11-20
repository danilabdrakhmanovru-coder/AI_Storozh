"""Evaluate earlier frozen suites without overwriting their historical results."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from app.contest import ContestRequest, inspect_request, read_policy
from experiment.three_domains.run import summary
from experiment.v7.evaluate import pairs

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def run(item):
    name, rows = item
    policy = read_policy()
    thresholds = yaml.safe_load((ROOT / "policy.yaml").read_bytes())["thresholds"]
    preds = []
    for row in rows:
        row.setdefault("is_dangerous", row["expected_decision"] == "block")
        preds.append(inspect_request(ContestRequest(user_prompt=row["user_prompt"], context=row["context"]),
                                     thresholds, policy))
    return name, summary(rows, preds)


if __name__ == "__main__":
    data = ROOT / "artifacts/astra_public"
    paths = {"public": data / "v3-holdout.jsonl", "v5": data / "v5-holdout.jsonl",
             "v6": OUT.parent / "v6/challenge.jsonl"}
    paths.update({s: OUT.parent / "three_domains" / (s + ".jsonl") for s in ("publishing", "agriculture", "sports")})
    datasets = [(name, [json.loads(l) for l in p.read_bytes().splitlines()]) for name, p in paths.items()]
    datasets.extend([("ferry", list(pairs())), ("radio", list(pairs("radio.txt", "radio")))])
    with ProcessPoolExecutor(max_workers=3) as pool:
        report = dict(pool.map(run, datasets))
    (OUT / "regression.json").write_text(json.dumps(report, indent=2), "utf8")
    for name, metric in report.items():
        print(name, metric["decision_correct"], metric["severe_block"], metric["matrix"])
