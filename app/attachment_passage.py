"""Offline passage classifier with offsets and learned context for reported speech."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re

import joblib
import numpy as np

DIR = Path(__file__).with_name('weights') / 'attachment-passage'
BOUNDARY = re.compile(r'(?:[\r\n]+|(?<=[.!?])\s+)')
SPACE = re.compile(r'\s+')
NUMBER = re.compile(r'\d+')

@dataclass(frozen=True)
class Passage:
    start: int
    end: int
    features: tuple

def compact(text):
    return SPACE.sub(' ', NUMBER.sub('0', text)).strip()

def passages(text):
    """Score every nonblank sentence/line. Retain adjacent text when the line contains a quotation."""
    start = 0
    for boundary in BOUNDARY.finditer(text):
        yield from cut(text, start, boundary.start())
        start = boundary.end()
    yield from cut(text, start, len(text))

def cut(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    while start < end:
        stop = min(end, start + 440)
        if stop < end:
            space = text.rfind(' ', start + 310, stop)
            if space > start:
                stop = space
        yield Passage(start, stop, (compact(text[start:stop]), compact(text[max(0,start-192):start]),
                                     compact(text[stop:stop+144])))
        if stop == end:
            return
        start = stop - 90

def focus(rows):
    return [r[0] for r in rows]

def neighbors(rows):
    return [r[1]+' [NEXT] '+r[2] for r in rows]

def preceding(rows):
    return [r[1][-145:] or '[empty]' for r in rows]

def following(rows):
    return [r[2][:112] or '[empty]' for r in rows]

def encoder_input(tokenizer, triples):
    # Truncate background before focus, in the same order used for fine-tuning.
    return tokenizer.encode_batch([(a[:430], b[-140:]+' [SEP] '+c[:100]) for a,b,c in triples])

class PassageHead:
    def __init__(self, directory, *, calibrating=False):
        from tokenizers import Tokenizer
        import onnxruntime as ort
        self.ranker = joblib.load(directory/'ranker.joblib')
        self.config = json.loads((directory/'config.json').read_bytes())
        self.attribution = joblib.load(directory/'attribution.joblib')
        rescue = self.config.get('rescue', {})
        self.prior = (joblib.load(directory.parent/'document-encoder'/'paired-text.joblib')
                      if calibrating or self.config.get('prior') or rescue.get('prior') else None)
        self.net = None
        if not calibrating and not self.config['neural'] and not rescue.get('neural'):
            return
        self.tok = Tokenizer.from_file(str(directory/'tokenizer.json'))
        self.tok.enable_truncation(max_length=128)
        self.tok.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.net = ort.InferenceSession(str(directory/'int8.onnx'), opts, providers=['CPUExecutionProvider'])

    def candidates(self, text):
        unique = {}
        for span in passages(text):
            unique.setdefault(span.features, span)
        spans = list(unique.values())
        if not spans:
            return []
        x = [s.features for s in spans]
        ranks = self.ranker.decision_function(x)
        top = np.argsort(-ranks,kind='stable')[:self.config['top_k']]
        encoded = encoder_input(self.tok,[x[i] for i in top]) if self.net else [None]*len(top)
        prior = self.prior.decision_function([x[i][0] for i in top]) if self.prior is not None else np.zeros(len(top))
        cited = self.attribution.decision_function([x[i] for i in top])
        out=[]
        for i, enc, old, quotation in zip(top,encoded,prior,cited):
            neural=0.
            if enc is not None:
                ids = np.array([enc.ids],dtype=np.int64)
                logits = self.net.run(None,{'input_ids':ids, 'attention_mask':np.ones_like(ids),
                                             'token_type_ids':np.array([enc.type_ids],dtype=np.int64)})[0][0]
                neural=float(logits[1]-logits[0])
            out.append((float(ranks[i]),neural,float(old),float(quotation),spans[i]))
        return out

    def old_support(self, attachment, candidates):
        from .attachment_intent import detect as previous_hit, load as previous, scores_for, WINDOW
        if len(attachment.content) <= WINDOW and previous_hit([attachment]):
            return np.ones(len(candidates),dtype=bool)
        old = previous()
        return scores_for(old['classifier'],[row[-1].features[0] for row in candidates]) >= old['threshold']

    def detect(self, attachments, candidates_by_index=None):
        from .attachment_intent import detect as previous_hit, WINDOW
        for index, attachment in enumerate(attachments):
            candidates = (self.candidates(attachment.content) if candidates_by_index is None
                          else candidates_by_index[index])
            scores = [(linear*self.config['linear']+neural*self.config['neural']+old*self.config['prior']
                       -self.config['quote_penalty']*max(0.,citation+.2),span)
                      for linear,neural,old,citation,span in candidates]
            if not scores:
                continue
            best,span = max(scores, key=lambda p:p[0])
            supported = (self.config['supported'] <= best < self.config['threshold']
                         and len(attachment.content) <= WINDOW and previous_hit([attachment]))
            if best >= self.config['threshold'] or supported:
                return dict(source='attachment',index=index,start=span.start,end=span.end,
                            rule='attachment_passage_supported' if supported else 'attachment_passage')
            rescue = self.config.get('rescue')
            if rescue:
                eligible = self.old_support(attachment,candidates)
                for allowed,(linear,neural,old,citation,span) in zip(eligible,candidates):
                    score = (rescue['linear']*linear + rescue['neural']*neural + rescue['prior']*old
                             - rescue['quote_penalty']*max(0.,citation+.2))
                    if allowed and score >= rescue['threshold']:
                        return dict(source='attachment',index=index,start=span.start,end=span.end,
                                    rule='attachment_passage_consensus')
        return None

@lru_cache(maxsize=1)
def load():
    return PassageHead(DIR)

def detect(attachments):
    return load().detect(attachments)

