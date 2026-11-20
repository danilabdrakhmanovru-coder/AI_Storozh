"""Evaluate complete gateway decisions, preserving paired results and diagnostic reasons."""
import argparse
import hashlib
import json

import yaml

from app.contest import ContestRequest, inspect_request, read_policy
from experiment.v10.benchmark import ROOT, holdout, scores
from experiment.v11.data import OUT, evaluation


def run(version, name, rows, out=OUT):
    thresholds = yaml.safe_load((ROOT / "policy.yaml").read_bytes())["thresholds"]
    policy = read_policy()
    results = []
    for row in rows:
        p = inspect_request(ContestRequest.model_validate(row), thresholds, policy)
        results.append(dict(id=row["id"], decision=p["decision"], risk_category=p["risk_category"],
                            reason=p["explanation"]))
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    report = dict(sha256=digest, scores=scores(rows, results), rows=results)
    (out / f"{version}-{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf8")
    print(name, report["scores"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=("v10", "v11"))
    parser.add_argument("sample", choices=("new", "previous", "public"))
    args = parser.parse_args()
    if args.sample == "public":
        rows = [json.loads(s) for s in (ROOT / "artifacts/astra_public/v3-holdout.jsonl").read_bytes().splitlines()]
    else:
        rows = evaluation() if args.sample == "new" else holdout("final")
    run(args.version, args.sample, rows)
