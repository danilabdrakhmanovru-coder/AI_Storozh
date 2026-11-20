"""Paired check on a fixed sample of external label-zero prompts; store no source text."""
import csv
import hashlib
import io
import json
import random
from unittest.mock import patch

import yaml

from app.contest import ContestRequest, inspect_request, read_policy
from experiment.external.run import HERE, SOURCE_HASHES, digest
from experiment.v9.evaluate import OUT, ROOT


def main():
    seen = {r["sha"] for r in json.loads((HERE / "manifest.json").read_bytes())["rows"]}
    thresholds = yaml.safe_load((ROOT / "policy.yaml").read_bytes())["thresholds"]
    policy = read_policy()
    results = []
    for source, digest_expected in SOURCE_HASHES.items():
        raw = (HERE / (source + ".csv")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest_expected:
            raise ValueError(f"Unexpected file content: {source}")
        population = []
        for index, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))):
            text = row["translation"].strip()
            key = digest(text.casefold())
            if row["label"] != "0" or not 0 < len(text) <= 10000 or key in seen or digest(text) in seen:
                continue
            seen.add(key)
            population.append((index, text))
        for index, text in random.Random("audit-v9-" + source).sample(population, 250):
            req = ContestRequest(user_prompt=text, context={"target_agent": "it_ops_agent", "user_role": "developer"})
            with patch("app.category_model.detects_jailbreak", return_value=False):
                before = inspect_request(req, thresholds, policy)["decision"]
            after = inspect_request(req, thresholds, policy)["decision"]
            results.append(dict(source=source, index=index, sha256=digest(text), v8=before, v9=after))
    report = dict(n=len(results), blocked_v8=sum(r["v8"] == "block" for r in results),
                  blocked_v9=sum(r["v9"] == "block" for r in results),
                  new_blocks=sum(r["v9"] == "block" and r["v8"] != "block" for r in results))
    print(report)
    (OUT / "external-control.json").write_text(json.dumps({**report, "rows": results}, indent=2), "utf8")


if __name__ == "__main__":
    main()
