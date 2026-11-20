"""Upload each unlabeled dataset to the running local contest API and compare verdicts."""
import gzip
import json
from urllib.request import Request, urlopen

from experiment.ten_domains.generate import OUT


def main():
    manifest = json.loads((OUT / "manifest.json").read_bytes())
    counts = {}
    for slug in manifest:
        rows = (OUT / "data" / (slug + ".jsonl")).read_bytes().splitlines()
        packets = []
        for line in rows:
            row = json.loads(line)
            packets.append({k: row[k] for k in ("id", "user_prompt", "context")})
        body = "\n".join(json.dumps(p, ensure_ascii=False) for p in packets).encode()
        req = Request("http://127.0.0.1:8001/v1/datasets/check?format=jsonl", data=body,
                      headers={"Content-Type": "application/octet-stream"})
        with urlopen(req, timeout=180) as reply:
            expected = int(reply.headers["X-Total-Rows"])
            results = [json.loads(line) for line in reply]
        saved = gzip.decompress((OUT / "v8" / (slug + ".jsonl.gz")).read_bytes()).splitlines()
        if not len(results) == expected == len(saved) == len(packets):
            raise ValueError(f"Incomplete response for {slug}")
        for packet, line, actual in zip(packets, saved, results):
            predicted = json.loads(line)
            if actual["id"] != packet["id"] or any(actual[k] != predicted[k] for k in
                                                    ("id", "decision", "risk_category", "risk_level")):
                raise ValueError(f"API verdict differs for {packet['id']}")
        counts[slug] = expected
        print(slug, expected, flush=True)
    (OUT / "api_verified.json").write_text(json.dumps(counts, indent=2), "utf8")


if __name__ == "__main__":
    main()
