"""Recompute paired metrics from frozen packets and saved CPU verdicts."""
import gzip
import json
from collections import Counter

from experiment.ten_domains.evaluate import read_rows
from experiment.ten_domains.generate import OUT
from experiment.three_domains.run import summary


def main():
    manifest = json.loads((OUT / "manifest.json").read_bytes())
    report = {}
    for version in ("v7", "v8"):
        groups = {name: ([], []) for name in ("development", "holdout", "total")}
        metrics = {}
        for slug, entry in manifest.items():
            rows = read_rows(OUT / "data" / (slug + ".jsonl"))
            path = OUT / version / (slug + ".jsonl.gz")
            preds = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
            if len(rows) != entry["size"] or [r["id"] for r in rows] != [p["id"] for p in preds]:
                raise ValueError(f"Incomplete predictions for {slug}")
            metrics[slug] = summary(rows, preds)
            for group in (entry["split"], "total"):
                groups[group][0].extend(rows)
                groups[group][1].extend(preds)
        for name, (rows, preds) in groups.items():
            metric = summary(rows, preds)
            for field in ("risk_category", "risk_level"):
                blocked = Counter()
                expected = Counter()
                for row, pred in zip(rows, preds):
                    if row["expected_decision"] == "block":
                        label = row[field]
                        expected[label] += 1
                        blocked[label] += int(pred["decision"] == "block")
                metric["block_by_" + field] = {label: {"blocked": blocked[label], "total": total}
                                                 for label, total in sorted(expected.items())}
            mistakes = Counter((r["risk_category"], p["risk_category"]) for r, p in zip(rows, preds)
                               if r["risk_category"] != p["risk_category"])
            metric["category_errors"] = [{"expected": a, "predicted": b, "count": n}
                                         for (a, b), n in mistakes.most_common()]
            metrics[name] = metric
        report[version] = metrics
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2), "utf8")
    for name in [*manifest, "development", "holdout", "total"]:
        for version in report:
            m = report[version][name]
            print(name, version, m["decision_correct"], m["category_correct"], m["level_correct"],
                  m["severe_block"], m["severe_total"], m["matrix"])


if __name__ == "__main__":
    main()
