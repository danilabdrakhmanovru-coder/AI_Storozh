"""Evaluate the public contest entry point locally. Persist one aligned verdict per input."""
import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from app.contest import ContestRequest, inspect_request, read_policy
from experiment.three_domains.run import summary
from experiment.ten_domains.generate import OUT


def read_rows(path):
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.splitlines()]


def evaluate(job):
    slug, entry, output = job
    path = OUT / "data" / (slug + ".jsonl")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {slug}")
    thresholds = yaml.safe_load((OUT.parents[1] / "policy.yaml").read_bytes())["thresholds"]
    policy = read_policy()
    rows = [json.loads(line) for line in raw.splitlines()]
    predictions = []
    for row in rows:
        req = ContestRequest(user_prompt=row["user_prompt"], context=row["context"])
        predictions.append(dict(id=row["id"], **inspect_request(req, thresholds, policy)))
    content = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions).encode()
    (output / (slug + ".jsonl.gz")).write_bytes(gzip.compress(content, mtime=0))
    return slug, summary(rows, predictions)


def main(version, workers, split):
    manifest = json.loads((OUT / "manifest.json").read_bytes())
    output = OUT / version
    output.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reports = {}
        jobs = [(slug, entry, output) for slug, entry in manifest.items() if split == "all" or entry["split"] == split]
        for slug, result in pool.map(evaluate, jobs):
            reports[slug] = result
            if manifest[slug]["split"] == "development":
                print(slug, result["decision_correct"], result["matrix"], flush=True)
    name = "metrics.json" if split == "all" else f"{split}-metrics.json"
    (output / name).write_text(json.dumps(reports, indent=2), "utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=["v7", "v8"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="all")
    args = parser.parse_args()
    main(args.version, args.workers, args.split)
