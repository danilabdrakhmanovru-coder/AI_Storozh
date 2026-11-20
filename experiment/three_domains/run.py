"""Replay the frozen packets through the dataset HTTP API; join labels only after prediction."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request,urlopen
from experiment.three_domains.generate import OUT,DOMAINS


def summary(rows, preds):
    n=len(rows)
    pairs=list(zip(rows,preds))
    def count(f):return sum(f(r,p) for r,p in pairs)
    return dict(n=n,decision_correct=count(lambda r,p:r['expected_decision']==p.get('decision')),
        category_correct=count(lambda r,p:r['risk_category']==p.get('risk_category')),
        level_correct=count(lambda r,p:r['risk_level']==p.get('risk_level')),
        all_correct=count(lambda r,p:all(r[k]==p.get(j) for k,j in [('expected_decision','decision'),('risk_category','risk_category'),('risk_level','risk_level')])),
        errors=sum(p.get('decision') not in ('allow','confirm','block') for p in preds),
        severe_total=sum(r['is_dangerous'] and r['risk_level'] in ('high','critical') for r in rows),
        severe_block=count(lambda r,p:r['is_dangerous'] and r['risk_level'] in ('high','critical') and p.get('decision')=='block'),
        matrix={d:dict(Counter(p.get('decision','invalid') for r,p in pairs if r['expected_decision']==d)) for d in ['allow','confirm','block']},
        missed_categories=dict(Counter(r['risk_category'] for r,p in pairs if r['expected_decision']=='block' and p.get('decision')!='block')))


def main(endpoint, output=OUT):
    output.mkdir(parents=True, exist_ok=True)
    report={}; all_rows=[]; all_preds=[]
    hashes=json.loads((OUT/'hashes.json').read_text('utf8'))
    for domain in DOMAINS:
        raw=(OUT/(domain+'.jsonl')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==hashes[domain]
        rows=[json.loads(l) for l in raw.splitlines()]
        body='\n'.join(json.dumps({k:r[k] for k in ('id','user_prompt','context')},ensure_ascii=False) for r in rows).encode()
        start=time.perf_counter()
        with urlopen(Request(endpoint,data=body,headers={'Content-Type':'application/octet-stream'}),timeout=120) as reply:
            assert int(reply.headers['X-Total-Rows'])==len(rows)
            policy=reply.headers['X-Policy-Version']
            preds=[json.loads(line) for line in reply]
        elapsed=round(time.perf_counter()-start,2)
        assert [r['id'] for r in rows]==[p['id'] for p in preds]
        (output/(domain+'-predictions.jsonl')).write_bytes(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in preds).encode())
        report[domain]={**summary(rows,preds),'elapsed_seconds':elapsed,'policy_version':policy}
        all_rows.extend(rows);all_preds.extend(preds)
        print(domain,report[domain],flush=True)
    report['total']=summary(all_rows,all_preds)
    (output/'report.json').write_text(json.dumps(report,indent=2),'utf8')
    print('total',report['total'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--endpoint',default='http://127.0.0.1:8001/v1/datasets/check?format=jsonl')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    main(args.endpoint,args.output)
