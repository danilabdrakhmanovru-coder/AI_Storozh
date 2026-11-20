"""Train on the public development partition and previously measured synthetic development data."""
import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion,make_pipeline
from sklearn.svm import LinearSVC
from sklearn.base import clone
from app.contest import ContestRequest
from app.intent_features import view,features

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def lines(path):
    return [json.loads(l) for l in path.read_text('utf-8-sig').splitlines()]


def main():
    hold=set(json.loads((ROOT/'experiment/v3/split.json').read_text())['holdout_ids'])
    rows=[r for r in lines(ROOT/'artifacts/astra_public/dev.jsonl') if r['id'] not in hold]
    # Use a single domain to avoid weighting five near-identical versions of each pattern.
    rows += [r for r in lines(ROOT/'experiment/synthetic1000/cases.jsonl') if r['tags'][0]=='bank' and '-' in r['tags'][1]]
    texts=[view(ContestRequest.model_validate(r)) for r in rows]
    categories=[r['risk_category'] for r in rows]
    danger=[r['expected_decision']=='block' for r in rows]
    for note in (OUT/'benign_context.txt').read_text('utf-8-sig').splitlines():
        for prompt in ('Сделай краткий обзор.', 'What are the main points?', 'Which items need attention?'):
            packet=ContestRequest.model_validate(dict(user_prompt=prompt,context=dict(
                target_agent='it_ops_agent',user_role='developer',attachments=[dict(content=note)])))
            texts.append(view(packet)); categories.append('none'); danger.append(False)
    extra=[]; extra_y=[]
    for file in ['train.txt','v2/additional.txt']:
        for line in (ROOT/'experiment'/file).read_text('utf-8-sig').splitlines():
            if '|' not in line: continue
            label,text=line.split('|',1)
            if label=='safe':
                texts.append(features(text)); categories.append('none'); danger.append(False)
            else:
                extra.append(features(text)); extra_y.append(True)
    for line in (OUT/'training.txt').read_text('utf-8-sig').splitlines():
        label,*examples=line.split('|')
        for text in examples:
            texts.append(features(text)); categories.append(label); danger.append(label!='none')
    pipeline=make_pipeline(FeatureUnion([
        ('word',TfidfVectorizer(sublinear_tf=True,ngram_range=(1,2))),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,sublinear_tf=True))
    ]),LinearSVC(class_weight='balanced',random_state=42))
    detector=clone(pipeline).fit(texts+extra,danger+extra_y)
    pipeline.fit(texts,categories)
    joblib.dump(dict(model=pipeline,danger=detector,min_margin=0.,danger_threshold=0.),OUT/'categories.joblib',compress=3)

if __name__=='__main__':main()
