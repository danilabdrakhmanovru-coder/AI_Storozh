"""Supplementary classifier-only coverage over both complete downloaded CSVs."""
import csv
import json
from collections import Counter
from app.classifier import load_classifier
from experiment.compare import decision, load
from experiment.external.run import HERE, SOURCES

def main():
    model = load_classifier()
    seen = {t.strip().casefold() for fn in ["train.txt","test.txt","v2/additional.txt","v2/test.txt"] for _,t in load(fn)}
    result = {}
    for source in SOURCES:
        texts, labels = [], []
        with (HERE/(source+".csv")).open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                text = row["translation"].strip()
                if not text or len(text)>10000 or text.casefold() in seen:
                    continue
                seen.add(text.casefold())
                texts.append(text)
                labels.append(int(row["label"]))
        counts = {"safe":Counter(),"threat":Counter()}
        for i in range(0,len(texts),128):
            scores = model.predict_proba(texts[i:i+128])[:,list(model.classes_).index("threat")]
            for label,score in zip(labels[i:],scores):
                counts["threat" if label else "safe"][decision(score)] += 1
        result[source] = counts
    (HERE/"full-local.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result)

if __name__=="__main__":
    main()
