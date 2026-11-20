"""Produce paired decisions and domain breakdown from the official evaluator output."""
import argparse
from collections import Counter

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(s) for s in path.read_text('utf-8-sig').splitlines()]


def summarize(raw_predictions):
    rows = read(OUT/'cases.jsonl')
    preds = read(raw_predictions)
    by_id = {p['id']: p for p in preds}
    assert len(by_id) == len(preds) == len(rows) == 1000
    fields = ('id', 'decision', 'risk_level', 'risk_category', 'explanation', 'latency_ms', '_error')
    compact = [{k: by_id[r['id']][k] for k in fields if k in by_id[r['id']]} for r in rows]
    (OUT/'predictions.jsonl').write_bytes(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in compact).encode())
    breakdown = {}
    for row, pred in zip(rows, compact):
        expected, actual = row['expected_decision'], pred.get('decision','invalid')
        for group in ('all', row['tags'][0], row['risk_level'], row['risk_category'], row['tags'][1].split('-')[0]):
            key = f'{group}/{expected}'
            breakdown.setdefault(key,Counter())[actual] += 1
    (OUT/'breakdown.json').write_text(json.dumps(breakdown,indent=2,sort_keys=True),'utf8')
    for k in sorted(breakdown):
        if k.split('/')[0] in {'all','bank','clinic','support','education','logistics','high','critical','history','attachment'}:
            print(k,dict(breakdown[k]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions',type=Path)
    summarize(parser.parse_args().predictions)

