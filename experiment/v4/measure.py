"""Paired decision/category diagnostics. No fitting or inference-time labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import yaml
from app.contest import ContestRequest,inspect_request,read_policy
from experiment.v4.train import ROOT,lines

def main(path,output):
    policy=read_policy()
    thresholds=yaml.safe_load((ROOT/'policy.yaml').read_bytes())['thresholds']
    rows=[]
    for r in lines(path):
        p=inspect_request(ContestRequest.model_validate(r),thresholds,policy)
        rows.append({**p,'id':r['id'],'expected':r['expected_decision'],'label':r['risk_category'],'severity':r['risk_level']})
    output.write_text(json.dumps(rows,ensure_ascii=False,indent=2),'utf8')
    print('N, decisions, categories',len(rows),sum(r['decision']==r['expected'] for r in rows),sum(r['risk_category']==r['label'] for r in rows))
    severe=[r for r in rows if r['expected']=='block' and r['severity'] in ('high','critical')]
    print('strict',sum(r['decision']=='block' for r in severe),len(severe))
    print(Counter((r['label'],r['risk_category']) for r in rows if r['label']!=r['risk_category']).most_common(16))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('data',type=Path)
    parser.add_argument('output',type=Path)
    a=parser.parse_args()
    main(a.data,a.output)
