"""Compare aligned verdicts, including safe requests newly blocked by either release."""
import gzip
import json
from collections import Counter

from experiment.v10.benchmark import DOMAINS, OUT, ROOT, fresh, holdout, read_lines, scores


def compare(rows, before, after):
    paired = zip(rows, before, after, strict=True)
    transitions = Counter()
    for row, a, b in paired:
        assert row["id"] == a["id"] == b["id"]
        if row["expected_decision"] == "allow":
            transitions[a["decision"] + " -> " + b["decision"]] += 1
    return {"v9": scores(rows, before), "v10": scores(rows, after), "safe_decisions": dict(transitions)}


def predictions(version, group):
    return read_lines(gzip.decompress((OUT / version / (group + ".jsonl.gz")).read_bytes()))


if __name__ == "__main__":
    result = {}
    for group in ("final", "holdout", "nine-control", "public"):
        if group == "public":
            rows = read_lines((ROOT / "artifacts/astra_public/v3-holdout.jsonl").read_bytes())
        elif group == "nine-control":
            rows = list(fresh("acceptance.txt", "acceptance"))
        else:
            rows = holdout(group)
        a, b = predictions("v9", group), predictions("v10", group)
        result[group] = compare(rows, a, b)
        if group == "final":
            for source in ("external", "authored"):
                selected = [i for i, r in enumerate(rows) if r["id"].startswith("external") == (source == "external")]
                result["final-" + source] = compare([rows[i] for i in selected], [a[i] for i in selected],
                                                    [b[i] for i in selected])
    rows, a, b = [], [], []
    for domain in json.loads((DOMAINS / "manifest.json").read_bytes()):
        rows.extend(read_lines((DOMAINS / "data" / (domain + ".jsonl")).read_bytes()))
        a.extend(predictions("v9", domain))
        b.extend(predictions("v10", domain))
    result["ten-domains"] = compare(rows, a, b)
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2), "utf8")
