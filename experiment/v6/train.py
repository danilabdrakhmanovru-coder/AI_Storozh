"""Refit v5's text classifiers using paired legitimate and malicious requests."""
from experiment.v5.train import OUT as V5
from experiment.v4.train import lines,main as fit
from experiment.v6.evaluate import OUT

def training_rows(path):
    rows=[]
    for line in path.read_text('utf-8-sig').splitlines():
        key,*examples=line.split('|')
        cat,_,source=key.partition('@')
        for text in examples:
            ctx=dict(target_agent='it_ops_agent',user_role='developer',history=[],attachments=[])
            if source=='history':
                ctx['history']=[dict(role='user',content=text)]
                text='Теперь продолжи эту тему.'
            if source=='attachment':
                ctx['attachments']=[dict(content=text)]
                text='Summarize the following material.'
            rows.append(dict(context=ctx,user_prompt=text,risk_category=cat,expected_decision='allow' if cat=='none' else 'block'))
    return rows


def main(additional=(), destination=OUT/'categories.joblib'):
    rows=[r for r in lines(V5/'museum.jsonl') if r['tags'][1]=='development'][:24]
    fit([*rows, *training_rows(OUT/'training.txt'), *additional], destination)

if __name__=='__main__':main()
