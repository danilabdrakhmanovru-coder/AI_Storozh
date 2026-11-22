"""Offline ONNX CPU inference. STOROZH_ENCODER_DIR selects the trained bundle."""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

CATEGORIES = ("none","sensitive_action","jailbreak","prompt_injection","indirect_injection",
              "system_prompt_extraction","system_prompt_override","tool_abuse","privilege_escalation",
              "unauthorized_data_access","data_exfiltration","harmful_content","fraud_social_engineering","abuse")
LEVELS = ("none","low","medium","high","critical")
MARKERS = ["[PROMPT]","[HISTORY]","[ATTACHMENT]"]
SIGNAL = re.compile(r"ignore|override|system|secret|admin|base64|парол|инструкц|привилег|правил|секрет|[A-Za-z0-9+/=]{24,}",re.I)


def excerpt(text):
    # Retain the beginning, a few non-overlapping evidence windows and the tail.
    text = str(text)
    for marker in MARKERS:
        text = text.replace(marker, marker.lower().replace('[','(').replace(']',')'))
    if len(text) <= 700:
        return text
    parts, end = [text[:160]], 160
    for m in SIGNAL.finditer(text):
        if m.start() < end:
            continue
        end = m.start()+100
        parts.append(text[max(0,m.start()-40):end])
        if len(parts) == 4:
            break
    return ' … '.join([*parts,text[-120:]])


def encode(tokenizer, prompt, context):
    # user_role and system_prompt do not enter the learned representation.
    # Reserve token space per source, independently of tokenizer padding.
    h = context.get("history", [])
    a = context.get("attachments", [])
    hist = [x for x in h if x.get("role") in ("user","assistant")][-4:]
    # Prioritise suspicious documents, while deterministic rules scan every document.
    docs = sorted(a, key=lambda x: not bool(SIGNAL.search(str(x.get("content","")))))[:3]
    sources = [[excerpt(prompt)], [x["role"]+" "+excerpt(x["content"]) for x in hist],
               [excerpt(x.get("type",""))[:20]+" "+excerpt(x.get("name",""))[:40]+" "+excerpt(x["content"]) for x in docs]]
    cls = tokenizer.token_to_id("[CLS]")
    ids = [cls if cls is not None else tokenizer.token_to_id("<s>")]
    for marker, parts, budget in zip(MARKERS,sources,(118,56,72)):
        ids.append(tokenizer.token_to_id(marker))
        for part in parts:
            pieces = tokenizer.encode(part,add_special_tokens=False).ids
            limit = budget//len(parts)
            if len(pieces)>limit:
                hit = SIGNAL.search(part)
                if hit:
                    left = limit//3
                    middle = limit//2
                    evidence = tokenizer.encode(part[hit.start():hit.start()+140],add_special_tokens=False).ids
                    selected = pieces[:left]+evidence[:middle]
                    pieces = selected+pieces[-(limit-len(selected)):]
                else:
                    front = limit*2//3
                    pieces = pieces[:front]+pieces[-(limit-front):]
            ids.extend(pieces)
    sep = tokenizer.token_to_id("[SEP]")
    ids.append(sep if sep is not None else tokenizer.token_to_id("</s>"))
    return ids


class Encoder:
    def __init__(self, directory, filename="model.int8.onnx"):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        directory = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(directory/"tokenizer.json"))
        self.config = json.loads((directory/"thresholds.json").read_text(encoding="utf8"))
        if not 0 <= self.config["confirm"] < self.config["block"] <= 1:
            raise ValueError("Encoder thresholds must satisfy 0 <= confirm < block <= 1")
        if self.config["categories"] != list(CATEGORIES) or self.config["levels"] != list(LEVELS):
            raise ValueError("Encoder label order mismatch")
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = min(8,os.cpu_count() or 1)
        opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory/filename),sess_options=opt,providers=["CPUExecutionProvider"])

    def evidence(self,user_prompt,context):
        ids = np.array([encode(self.tokenizer,user_prompt,context)],dtype=np.int64)
        category,danger,level = self.session.run(None,{"input_ids":ids,"attention_mask":np.ones_like(ids)})
        if not all(np.isfinite(v).all() for v in (category,danger,level)):
            raise ValueError("Non-finite encoder output")
        attack_indices = range(2,len(CATEGORIES))
        alternate = CATEGORIES[max(attack_indices,key=lambda i: category[0,i])]
        return {"category":CATEGORIES[int(category.argmax())],"alternative":alternate,"risk_level":LEVELS[int(level.argmax())],
                "danger_prob":float(1/(1+np.exp(-np.clip(float(danger.item()),-60,60))))}

    def predict(self,user_prompt,context):
        prediction = self.evidence(user_prompt,context)
        prediction.pop("alternative")
        return prediction


@lru_cache(maxsize=1)
def configured(directory):
    return Encoder(directory)


def predict(user_prompt,context):
    return configured(os.environ["STOROZH_ENCODER_DIR"]).predict(user_prompt,context)

