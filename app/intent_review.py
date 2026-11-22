"""Local CPU review of learned decisions. Policy operations are handled before this layer."""
import json
import os
from functools import lru_cache
from pathlib import Path
import zipfile

import joblib
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .semantic_review import Reviewer
from .source_review import columns

WEIGHTS = Path(__file__).with_name('weights') / 'intent-review'
INJECTION = {'jailbreak', 'prompt_injection', 'system_prompt_extraction', 'system_prompt_override'}


class Detector:
    def __init__(self, graph, tokenizer):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4 if os.getenv('STOROZH_ATTACHMENT_INTENT')=='13' else 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(graph, opts, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(str(tokenizer))

    def outputs(self, text):
        tokens = self.tokenizer.encode(text).ids
        # Keep complete short inputs and bounded head/tail context on a single CPU call.
        if len(tokens) > 384:
            tokens = tokens[:240] + tokens[-144:]
        ids = np.asarray([tokens], dtype=np.int64)
        outputs = self.session.run(None, {'input_ids':ids, 'attention_mask':np.ones_like(ids)})
        logits = outputs[0][0]
        return float(np.clip(logits[1] - logits[0], -18, 18)), outputs[1][0] if len(outputs)>1 else None


class Features:
    def __init__(self, detector, local, semantic=(), pooled=()):
        self.detector = detector
        self.local = local
        self.semantic = semantic
        self.pooled = pooled
        self.previous = Reviewer()

    def extract(self, texts):
        vectors = self.previous.embedder.encode(texts, batch_size=1)
        heads = self.previous.model
        old = columns(vectors, texts, heads)
        prior = heads['calibration'].decision_function(old)
        local = self.local.decision_function(texts)
        results = [self.detector.outputs(t) for t in texts]
        specialized = np.array([r[0] for r in results])
        scores = np.column_stack((specialized, prior, local, old))
        if self.semantic:
            scores = np.column_stack([scores, *(h.decision_function(vectors) for h in self.semantic)])
        if self.pooled:
            representations=np.stack([r[1] for r in results])
            scores=np.column_stack([scores, *(h.decision_function(representations) for h in self.pooled),representations])
        cats = heads['subtype'].predict(vectors)
        return scores, [heads['categories'][int(i)] for i in cats]


def assess(model, features, prior_category):
    score = float(model['head'].predict_proba(features.reshape(1, -1))[0, 1])
    local_classes = model['local_classes']
    local = features[2:2+len(local_classes)]
    local_category = local_classes[int(np.argmax(local))]
    category = None
    if score >= model['upper']:
        # A supervised policy category supplies attribution whenever available.
        if local_category != 'none' and max(local) >= model['local_margin']:
            category = local_category
        elif features[0] >= 0 and prior_category not in INJECTION:
            category = 'prompt_injection'
        else:
            category = prior_category
    clear = score <= model['lower']
    clear_content = (model.get('clear_content', False) and score <= min(.15, model['lower'])
                     and local_category == 'none' and features[0] < 0)
    return clear, category, clear_content


class IntentReviewer:
    def __init__(self):
        meta = json.loads((WEIGHTS/'encoder.json').read_bytes())
        parts = [bytes_from_part(WEIGHTS / name) for name in meta['parts']]
        detector = Detector(b''.join(parts), WEIGHTS/'tokenizer.json')
        self.model = joblib.load(WEIGHTS/'calibration.joblib')
        semantic = joblib.load(WEIGHTS/'semantic.joblib') if self.model['extra'] else ()
        pooled = joblib.load(WEIGHTS/'pooled.joblib') if self.model.get('pooled') else ()
        self.features = Features(detector, joblib.load(WEIGHTS/'categories.joblib'), semantic, pooled)

    def assess(self, text):
        clear, category, _ = self.review(text)
        return clear, category

    def review(self, text):
        scores, cats = self.features.extract([text])
        return assess(self.model, scores[0], cats[0])


def bytes_from_part(path):
    with zipfile.ZipFile(path) as archive:
        return archive.read('chunk')


@lru_cache(maxsize=1)
def reviewer():
    return IntentReviewer()
