"""CPU-only classifier of live directives in untrusted attachments."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name('weights') / 'document-encoder'


class DocumentEncoder:
    def __init__(self, directory):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        config = json.loads((directory / 'config.json').read_bytes())
        self.threshold = config['threshold']
        self.prior_cut = config.get('prior_cut', float('inf'))
        self.supported_threshold = config.get('supported_threshold', self.threshold)
        self.coefficients = np.asarray(config.get('coef', [1, 0, 0]), dtype=float)
        if (self.coefficients.shape != (3,) or not np.isfinite(self.coefficients).all()
                or not np.isfinite([self.threshold, self.supported_threshold]).all() or not self.coefficients.any()):
            raise ValueError('Некорректная калибровка модели вложений')
        self.paired_model = None
        if self.coefficients[1]:
            import joblib
            self.paired_model = joblib.load(directory / 'paired-text.joblib')
        self.previous_model = None
        if self.coefficients[2] or 'prior_cut' in config:
            from .attachment_intent import load as previous
            self.previous_model = previous()['classifier']
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=config['window_tokens'], stride=config['overlap_tokens'])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        # The raw encoder is also used by the offline calibration stage.
        self.session = ort.InferenceSession(str(directory / 'model.int8.onnx'), options,
                                            providers=['CPUExecutionProvider'])

    def windows(self, text):
        if text.strip():
            first = self.tokenizer.encode(text)
            for section in (first, *first.overflowing):
                spans = [offset for offset in section.offsets if offset[1] > offset[0]]
                if spans:
                    yield section.ids, spans[0][0], spans[-1][1]

    def scores(self, windows):
        # One window per inference keeps dynamic quantization independent of other documents.
        for ids, start, end in windows:
            tensor = np.asarray([ids], dtype=np.int64)
            logits = self.session.run(None, {'input_ids': tensor, 'attention_mask': np.ones_like(tensor)})[0][0]
            yield float(logits[1] - logits[0]), start, end

    def detect(self, attachments):
        for index, attachment in enumerate(attachments):
            text = attachment.content
            windows = list(self.windows(text))
            for offset in range(0, len(windows), 8):
                batch = windows[offset:offset + 8]
                parts = [text[start:end] for _, start, end in batch]
                margins = np.zeros(len(batch))
                if self.coefficients[0]:
                    margins += self.coefficients[0] * np.array([score for score, _, _ in self.scores(batch)])
                thresholds = np.full(len(batch), self.threshold)
                if self.paired_model is not None:
                    margins += self.coefficients[1] * self.paired_model.decision_function(parts)
                if self.previous_model is not None:
                    previous = self.previous_model.decision_function(parts)
                    margins += self.coefficients[2] * previous
                    thresholds[previous >= self.prior_cut] = self.supported_threshold
                for margin, threshold, (_, start, end) in zip(margins, thresholds, batch):
                    if margin >= threshold:
                        return dict(source='attachment', index=index, start=start, end=end, rule='document_encoder')
        return None


@lru_cache(maxsize=1)
def load():
    return DocumentEncoder(WEIGHTS)


def detect(attachments):
    return load().detect(attachments)

