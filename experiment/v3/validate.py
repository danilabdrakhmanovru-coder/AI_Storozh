"""Compare paired verdicts on withheld examples, using identical expected labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import yaml
from sklearn.metrics import f1_score
from app.category_model import MODEL, load, packet_text
from app.contest import ContestRequest, inspect_request, read_policy


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]


def summarize(rows, predictions):
    labels = [r['risk_category'] for r in rows]
    cats = [p['risk_category'] for p in predictions]
    matrix = {d: dict(Counter(p['decision'] for r, p in zip(rows, predictions) if r['expected_decision'] == d))
              for d in ['allow', 'confirm', 'block']}
    severe = [p for r, p in zip(rows, predictions) if r['expected_decision']=='block' and r['risk_level'] in {'high','critical'}]
    return {'count': len(rows), 'category_correct': sum(a == b for a,b in zip(labels,cats)),
            'category_macro_f1': f1_score(labels,cats,labels=sorted(set(labels)),average='macro',zero_division=0),
            'matrix': matrix, 'severe_blocked': sum(p['decision']=='block' for p in severe), 'severe_total':len(severe)}


def main(data, previous):
    ids = set(json.loads(MODEL.with_name('split.json').read_text())['holdout_ids'])
    rows = [row for row in read_lines(data) if row['id'] in ids]
    baseline = {p['id']: p for p in read_lines(previous)}
    packets = [ContestRequest.model_validate(r) for r in rows]
    thresholds = yaml.safe_load((MODEL.parents[2]/'policy.yaml').read_bytes())['thresholds']
    policy = read_policy()
    new = [inspect_request(p,thresholds,policy) for p in packets]
    labels = [r['risk_category'] for r in rows]
    cats = load()["model"].predict([packet_text(p) for p in packets])
    report = {'baseline': summarize(rows,[baseline[r['id']] for r in rows]), 'updated': summarize(rows,new),
              'model_alone_category_correct':sum(int(a==b) for a,b in zip(labels,cats)),
              'category_correct_by_class':{label:[sum(r['risk_category']==label and p['risk_category']==label
                 for r,p in zip(rows,new)),labels.count(label)] for label in sorted(set(labels))}}
    print(json.dumps(report,indent=2))
    MODEL.with_name('validation.json').write_text(json.dumps(report,indent=2), 'utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('data',type=Path)
    parser.add_argument('previous_predictions',type=Path)
    args=parser.parse_args()
    main(args.data,args.previous_predictions)

