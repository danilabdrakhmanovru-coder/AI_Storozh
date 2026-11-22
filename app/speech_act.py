"""Offline speech-act inference; only the words of the utterance enter the model."""
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import joblib

WEIGHTS=Path(__file__).with_name("weights")
INFO={"educational","incident_description","question_about_process"}
SPLIT=re.compile(r"(?:[;!?]\s+|\b(?:а потом|а затем|затем|после этого|and then|after that|then)\b)",re.I)
FOLLOWUP=re.compile(r"(?:[.!?;]\s+|\b(?:а потом|а затем|затем|после этого|and then|after that|then|and|и)\s+)([^.!?;]+)",re.I)


@lru_cache(maxsize=2)
def linear(path):
    return joblib.load(path)


@lru_cache(maxsize=1)
def encoder():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    root=Path(os.getenv('STOROZH_SPEECH_ENCODER_PATH',WEIGHTS/'speech-encoder'))
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=2
    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=128)
    return tokenizer,ort.InferenceSession(str(root/'model.int8.onnx'),opts,providers=['CPUExecutionProvider']),json.loads((root/'labels.json').read_bytes())


def speech(text):
    # Consider the end separately, even for a long educational opening.
    samples=[text[:4096],*(m[1].strip()[:1024] for m in FOLLOWUP.finditer(text) if len(m[1].split())>1)]
    samples=list(dict.fromkeys(samples[:8]+samples[-2:]))
    if os.getenv("STOROZH_VARIANT","B")=="C":
        import numpy as np
        tokenizer,session,labels=encoder()
        acts=[]
        for sample in samples:
            ids=tokenizer.encode(sample).ids
            tensor=np.asarray([ids],dtype=np.int64)
            acts.append(labels[session.run(None,{'input_ids':tensor,'attention_mask':np.ones_like(tensor)})[0].argmax()])
    else:
        model=linear(os.getenv('STOROZH_SPEECH_PATH',str(WEIGHTS/'speech.joblib')))
        acts=list(model.predict(samples))
    for priority in ("bypass_attempt","execute_request"):
        if priority in acts:return priority
    return str(acts[0])

