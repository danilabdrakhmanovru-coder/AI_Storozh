"""Paired strict-block metrics for held-out new text, historical domains and public data."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import yaml

from app.contest import ContestRequest, inspect_request, read_policy

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DOMAINS = OUT.parent / "ten_domains"


def fresh(filename="final.txt", prefix="fresh"):
    for i, line in enumerate((OUT / filename).read_text("utf-8").splitlines()):
        group, text = line.split("|", 1)
        context = dict(target_agent="it_ops_agent", user_role=("developer", "sre", "admin")[i % 3])
        english = text.isascii() or not any("А" <= c <= "я" for c in text)
        if i % 4 == 1:
            context["history"] = [dict(role="user", content=text)]
            text = "Please continue." if english else "Продолжи."
        elif i % 4 == 2:
            context["history"] = [dict(role="user", content="Hello!" if english else "Здравствуйте!"),
                                  dict(role="assistant", content="How can I help?" if english else "Чем помочь?")]
        yield dict(id=f"{prefix}-{i+1:03}", user_prompt=text, context=context, group=group,
                   expected_decision="block" if group == "jailbreak" else "allow",
                   risk_category="jailbreak" if group == "jailbreak" else "none")


def metrics(rows, preds):
    assert [r["id"] for r in rows] == [p["id"] for p in preds]
    counts = Counter()
    for r, p in zip(rows, preds):
        decision = p["decision"]
        assert decision in {"allow", "confirm", "block"}
        if r["risk_category"] == "jailbreak":
            counts["jailbreak"] += 1
            counts["blocked"] += decision == "block"
            counts["correct_jailbreak"] += decision == "block" and p["risk_category"] == "jailbreak"
        if r["expected_decision"] == "allow":
            counts["benign"] += 1
            counts["false_block"] += decision == "block"
            counts["false_confirm"] += decision == "confirm"
            if "group" in r:
                counts[r["group"] + "_total"] += 1
                counts[r["group"] + "_blocked"] += decision == "block"
        counts["decision_correct"] += decision == r["expected_decision"]
        counts["category_correct"] += p["risk_category"] == r["risk_category"]
        counts["n"] += 1
    return dict(counts)


def run(job):
    name, rows, version = job
    if version == "v8" and name in json.loads((DOMAINS / "manifest.json").read_bytes()):
        lines = gzip.decompress((DOMAINS / "v8" / (name + ".jsonl.gz")).read_bytes()).splitlines()
        preds = [json.loads(line) for line in lines]
    else:
        thresholds = yaml.safe_load((ROOT / "policy.yaml").read_bytes())["thresholds"]
        policy = read_policy()
        preds = []
        from contextlib import nullcontext
        # v9 preserves the frozen v8 heads and adds one binary classifier.
        # Disable that additional head to replay archived v8 semantics.
        baseline = patch("app.category_model.detects_jailbreak", return_value=False) if version == "v8" else nullcontext()
        with baseline:
            for r in rows:
                req = ContestRequest(user_prompt=r["user_prompt"], context=r["context"])
                verdict = inspect_request(req, thresholds, policy)
                preds.append(dict(id=r["id"], decision=verdict["decision"], risk_category=verdict["risk_category"]))
    if name in {"fresh", "context", "public"}:
        (OUT / f"{version}-{name}-predictions.json").write_text(json.dumps(preds, indent=2), "utf8")
    return name, metrics(rows, preds)


def main(version, part):
    jobs = []
    manifest = json.loads((DOMAINS / "manifest.json").read_bytes())
    if part in {"all", "legacy"}:
        for name, entry in manifest.items():
            raw = (DOMAINS / "data" / (name + ".jsonl")).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
            jobs.append((name, [json.loads(l) for l in raw.splitlines()], version))
        public = ROOT / "artifacts/astra_public/v3-holdout.jsonl"
        jobs.append(("public", [json.loads(l) for l in public.read_bytes().splitlines()], version))
    if part in {"all", "fresh"}:
        jobs.append(("fresh", list(fresh()), version))
        jobs.append(("context", list(fresh("context-check.txt", "context")), version))
    with ProcessPoolExecutor(max_workers=3) as pool:
        result = dict(pool.map(run, jobs))
    if part in {"all", "legacy"}:
        for split in ("all_domains", "holdout"):
            agg = Counter()
            for name, entry in manifest.items():
                if split == "all_domains" or entry["split"] == split:
                    agg.update(result[name])
            result[split] = dict(agg)
    (OUT / f"{version}-{part}.json").write_text(json.dumps(result, indent=2), "utf8")
    print({k: v for k, v in result.items() if k in {"all_domains", "holdout", "public", "fresh", "context"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=["v8", "v9"])
    parser.add_argument("part", choices=["all", "legacy", "fresh"])
    args = parser.parse_args()
    main(args.version, args.part)
