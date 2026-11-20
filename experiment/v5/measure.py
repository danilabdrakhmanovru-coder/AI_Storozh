"""Measure all three metrics on identical rows, with labels withheld from inference."""
import argparse
import json
from pathlib import Path
from collections import Counter
import yaml
from app.contest import inspect_request,ContestRequest,read_policy
from experiment.v4.train import ROOT,lines
from experiment.v5.generate import OUT,DOMAINS


def measure(path):
    thresholds=yaml.safe_load((ROOT/'policy.yaml').read_bytes())['thresholds']
    policy=read_policy()
    details=[]
    sets=[(domain,OUT/(domain+'.jsonl')) for domain,*_ in DOMAINS]+[
        ('public-dev',ROOT/'artifacts/astra_public/v3-holdout.jsonl'),('v4',ROOT/'experiment/v4/challenge.jsonl')]
    for domain,data in sets:
        for row in lines(data):
            packet=ContestRequest(user_prompt=row['user_prompt'],context=row['context'])
            p=inspect_request(packet,thresholds,policy)
            split=row['tags'][1] if domain not in ('v4','public-dev') else 'previous'
            details.append(dict(domain=domain,split=split,id=row['id'],decision=p['decision'],category=p['risk_category'],
                level=p['risk_level'],want=row['expected_decision'],cat=row['risk_category'],severity=row['risk_level']))
    path.write_text(json.dumps(details,ensure_ascii=False),'utf8')
    for domain,_ in sets:
        for split in ('development','holdout','previous'):
            rows=[d for d in details if d['domain']==domain and d['split']==split]
            if rows:print(domain,split,len(rows),*[sum(d[a]==d[b] for d in rows) for a,b in [('decision','want'),('category','cat'),('level','severity')]],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);measure(parser.parse_args().output)
