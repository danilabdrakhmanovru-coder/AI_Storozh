"""Check every frozen evaluation packet through the running local batch API."""
import json
from urllib.request import Request, urlopen

from experiment.v11.data import OUT, evaluation


def check(rows, reference):
    packets = [{key: row[key] for key in ("id", "user_prompt", "context")} for row in rows]
    data = "\n".join(json.dumps(p, ensure_ascii=False) for p in packets).encode()
    uri = "http://127.0.0.1:8001/v1/datasets/check?format=jsonl"
    request = Request(uri, data=data, headers={"Content-Type": "application/octet-stream"})
    with urlopen(request, timeout=180) as response:
        assert int(response.headers["X-Total-Rows"]) == len(rows)
        answers = list(map(json.loads, response))
    ref = json.loads(reference.read_bytes())
    for actual, expected in zip(answers, ref["rows"], strict=True):
        for key in ("id", "decision", "risk_category"):
            assert actual[key] == expected[key], (expected["id"], key)
    print(f"Matched id, decision and category on {len(answers)} packets.")


if __name__ == "__main__":
    check(evaluation(), OUT / "v11-new.json")
