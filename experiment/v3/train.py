"""Train the CPU category model. Run python -m experiment.v3.train DATA.jsonl."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.base import clone
import yaml

from app.category_model import MODEL, packet_text
from app.contest import ContestRequest


def groups(texts):
    # Group related variants transitively, independent of labels or user IDs.
    stripped = [re.sub(r'https?://\S+|\d+', ' VALUE ', text) for text in texts]
    matrix = TfidfVectorizer(analyzer='char', ngram_range=(3, 4)).fit_transform(stripped)
    return connected_components(cosine_similarity(matrix) >= .48, directed=False)[1]


def main(path):
    rows = [json.loads(line) for line in path.read_text('utf-8-sig').splitlines()]
    texts = [packet_text(ContestRequest.model_validate(row)) for row in rows]
    labels = np.array([row['risk_category'] for row in rows])
    families = groups(texts)
    train, test = next(StratifiedGroupKFold(4, shuffle=True, random_state=42).split(texts, labels, families))
    assert not set(families[test]) & set(families[train])
    model = make_pipeline(FeatureUnion([
        ('word', TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))),
        ('char', TfidfVectorizer(analyzer='char', sublinear_tf=True, ngram_range=(2, 5), min_df=2))
    ]), LinearSVC(class_weight='balanced', random_state=42))
    blocked = yaml.safe_load((MODEL.parents[2]/'security_policy.yaml').read_bytes())['always_block']
    benign_margins = []
    for fit, val in StratifiedGroupKFold(3, shuffle=True, random_state=17).split(train, labels[train], families[train]):
        fold = clone(model).fit([texts[i] for i in train[fit]], labels[train[fit]])
        for idx, scores in zip(train[val], fold.decision_function([texts[i] for i in train[val]])):
            if labels[idx] != 'none':
                continue
            order = np.argsort(scores)
            margin = scores[order[-1]] - scores[order[-2]]
            benign_margins.append(margin if fold.classes_[order[-1]] in blocked else 0.)
    cutoff = float(np.quantile(benign_margins, .97, method='higher')) + 1e-8
    model.fit([texts[i] for i in train], labels[train])
    joblib.dump({'model':model, 'min_margin':cutoff}, MODEL, compress=3)
    metadata = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'train_counts': dict(Counter(labels[train])), 'test_counts': dict(Counter(labels[test])),
                'groups': len(set(families)), 'holdout_ids': [rows[i]['id'] for i in test]}
    MODEL.with_name('split.json').write_text(json.dumps(metadata, indent=2), 'utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    main(parser.parse_args().data)
