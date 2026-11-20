"""Check the local batch endpoint on the same unlabeled packets as offline evaluation."""
import argparse
import gzip
import hashlib
import json
from urllib.request import Request, urlopen

from experiment.v10.benchmark import OUT, holdout, read_lines, scores


def check(version, group, record):
    rows = holdout(group)
    packets = [{k: r[k] for k in ("id", "user_prompt", "context")} for r in rows]
    payload = "\n".join(json.dumps(p, ensure_ascii=False) for p in packets).encode()
    endpoint = "http://127.0.0.1:8001/v1/datasets/check?format=jsonl"
    with urlopen(Request(endpoint, data=payload, headers={"Content-Type": "application/octet-stream"}), timeout=180) as res:
        assert int(res.headers["X-Total-Rows"]) == len(rows)
        preds = [{k: p[k] for k in ("id", "decision", "risk_category")} for p in map(json.loads, res)]
    metrics = scores(rows, preds)
    path = OUT / version / (group + ".jsonl.gz")
    if record:
        path.write_bytes(gzip.compress("".join(json.dumps(p) + "\n" for p in preds).encode(), mtime=0))
    else:
        assert read_lines(gzip.decompress(path.read_bytes())) == preds
    metrics["packets_sha256"] = hashlib.sha256(payload).hexdigest()
    (OUT / version / (group + "-api.json")).write_text(json.dumps(metrics, indent=2), "utf8")
    print({k: v for k, v in metrics.items() if k != "category_errors"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("version", choices=["v9", "v10"])
    p.add_argument("group", choices=["holdout", "final"])
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    check(args.version, args.group, args.record)
