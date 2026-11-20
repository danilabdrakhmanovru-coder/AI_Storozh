"""Replay the same labelled domain scenarios against a running contest endpoint."""
import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.request import Request, urlopen


def run(endpoint):
    rows = []
    for line in Path(__file__).with_name("cases.txt").read_text("utf-8-sig").splitlines():
        domain, split, expected, role, prompt = line.split("|", 4)
        packet = {"context": {"target_agent": "it_ops_agent", "user_role": role}, "user_prompt": prompt}
        request = Request(endpoint, data=json.dumps(packet).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            answer = json.load(response)
        rows.append(dict(domain=domain, split=split, expected=expected, prompt=prompt,
                         decision=answer["decision"], reason=answer["explanation"]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = run(args.endpoint)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    print(Counter((r["domain"], r["expected"], r["decision"]) for r in rows))
