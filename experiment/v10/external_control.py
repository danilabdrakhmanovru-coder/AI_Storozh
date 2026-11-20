"""Replay the 500 historical external controls and compare each decision with saved v9."""
import csv
import hashlib
import io
import json
from collections import Counter

import yaml

from app.contest import ContestRequest, inspect_request, read_policy
from experiment.v10.benchmark import OUT, ROOT, HERE, SOURCE_HASHES


if __name__ == "__main__":
    previous = json.loads((OUT.parent / "v9/external-control.json").read_bytes())
    paired = Counter()
    policy = read_policy()
    thresholds = yaml.safe_load((ROOT / "policy.yaml").read_bytes())["thresholds"]
    for source, checksum in SOURCE_HASHES.items():
        raw = (HERE / (source + ".csv")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == checksum
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        for row in previous["rows"]:
            if row["source"] != source:
                continue
            text = rows[row["index"]]["translation"].strip()
            assert hashlib.sha256(text.encode()).hexdigest() == row["sha256"]
            req = ContestRequest(user_prompt=text, context={"target_agent": "it_ops_agent", "user_role": "developer"})
            verdict = inspect_request(req, thresholds, policy)
            paired[row["v9"] + " -> " + verdict["decision"]] += 1
    (OUT / "external-control.json").write_text(json.dumps(paired, indent=2), "utf8")
    print(paired)
