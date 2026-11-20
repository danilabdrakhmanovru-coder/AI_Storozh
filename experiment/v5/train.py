"""Train categorical intent and conditional severity from development partitions only."""
from pathlib import Path
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from experiment.v4.train import main as train_categories, ROOT, lines
from app.intent_features import view
from app.contest import ContestRequest

OUT=Path(__file__).resolve().parent


def main():
    # One sector per development wording; exclude disputed text-only role labels.
    dev=[r for r in lines(OUT/'museum.jsonl') if r['tags'][1]=='development']
    train_categories(dev[:24], OUT/'categories.joblib')
    hold=set(json.loads((ROOT/'experiment/v3/split.json').read_text())['holdout_ids'])
    samples=[r for r in lines(ROOT/'artifacts/astra_public/dev.jsonl') if r['id'] not in hold] + dev
    models={}
    for cat in sorted({r['risk_category'] for r in samples}):
        subset=[r for r in samples if r['risk_category']==cat]
        levels=[r['risk_level'] for r in subset]
        if len(set(levels))==1:
            models[cat]=levels[0]
        else:
            pipeline=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),sublinear_tf=True),
                                   LinearSVC(class_weight='balanced',random_state=42))
            models[cat]=pipeline.fit([view(ContestRequest.model_validate(r)) for r in subset],levels)
    joblib.dump(models, OUT/'severity.joblib',compress=3)

if __name__=='__main__':main()
