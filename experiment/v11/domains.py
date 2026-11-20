"""Paired 1000-request regression across ten archived domains, sampled before inference."""
import gzip
import hashlib
import json
import random
from collections import Counter

import yaml

from app.contest import ContestRequest, inspect_request, read_policy
from experiment.v10.benchmark import DOMAINS, ROOT, read_lines
from experiment.v11.data import OLD, OUT


def main(version="v11", out=OUT):
    rng = random.Random(11038)
    thresholds = yaml.safe_load((ROOT / "policy.yaml").read_bytes())["thresholds"]
    policy = read_policy()
    versions = {"v10": Counter(), version: Counter()}
    changes = []
    for domain, entry in sorted(json.loads((DOMAINS / "manifest.json").read_bytes()).items()):
        raw = (DOMAINS / "data" / (domain + ".jsonl")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        rows = read_lines(raw)
        old = read_lines(gzip.decompress((OLD / "v10" / (domain + ".jsonl.gz")).read_bytes()))
        for i in sorted(rng.sample(range(len(rows)), 100)):
            row = rows[i]
            assert row["id"] == old[i]["id"]
            new = inspect_request(ContestRequest.model_validate(row), thresholds, policy)
            for name, p in [("v10", old[i]), (version, new)]:
                c = versions[name]
                c["right_category"] += p["risk_category"] == row["risk_category"]
                c["right_decision"] += p["decision"] == row["expected_decision"]
                c["unsafe_allowed"] += p["decision"] == "allow" and row["expected_decision"] == "block"
                c["false_blocks"] += p["decision"] == "block" and row["expected_decision"] == "allow"
            if any(old[i][k] != new[k] for k in ("risk_category", "decision")):
                changes.append(dict(domain=domain, id=row["id"], expected=row["risk_category"],
                                    before=old[i]["risk_category"], after=new["risk_category"]))
    (out / "domains.json").write_text(json.dumps(dict(count=1000, versions=versions, changes=changes), indent=2))
    print(versions)


if __name__ == "__main__":
    main()
