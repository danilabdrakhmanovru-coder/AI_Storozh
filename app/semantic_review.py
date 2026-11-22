"""Local cross-source classifier. Decisions and ACLs remain in the contest gateway."""
from functools import lru_cache
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np

DIR=Path(__file__).with_name('weights')/'semantic-review'
INJECTION={'jailbreak','prompt_injection','system_prompt_override','system_prompt_extraction'}
def lower(text):
    return text.casefold()

class Embedder:
    def __init__(self,onnx,tokenizer,threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer=Tokenizer.from_file(str(tokenizer))
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=threads
        opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(onnx,opts,providers=['CPUExecutionProvider'])
    def tokens(self,text):
        if len(text)>4200:
            mid=len(text)//2
            text=text[:1800]+' '+text[mid-350:mid+350]+' '+text[-1600:]
        ids=self.tokenizer.encode('query: '+text).ids
        if len(ids)>128:
            mid=len(ids)//2
            ids=ids[:70]+ids[mid-8:mid+8]+ids[-42:]
        return ids
    def encode(self,texts,batch_size=12):
        ids=[self.tokens(s) for s in texts]
        order=sorted(range(len(ids)),key=lambda k:len(ids[k]))
        result=np.empty((len(texts),384),dtype=np.float32)
        for start in range(0,len(order),batch_size):
            batch=order[start:start+batch_size]
            x=np.ones((len(batch),max(len(ids[k]) for k in batch)),dtype=np.int64)
            mask=np.zeros_like(x)
            for j,k in enumerate(batch):
                x[j,:len(ids[k])]=ids[k]
                mask[j,:len(ids[k])]=1
            hidden=self.session.run(None,dict(input_ids=x,attention_mask=mask,token_type_ids=np.zeros_like(x)))[0]
            pooled=(hidden*mask[:,:,None].astype(np.float32)).sum(axis=1)
            result[batch]=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-8)
        return result

class Reviewer:
    def __init__(self):
        self.config=json.loads((DIR/'config.json').read_bytes())
        self.stacked=self.config.get('architecture')=='stack'
        self.model=joblib.load(DIR/('stack.joblib' if self.stacked else 'heads.joblib'))
        self.embedder=None
        if self.config['semantic_weight']>0:
            with zipfile.ZipFile(DIR/'encoder.zip') as z:
                self.embedder=Embedder(z.read('model.onnx'),DIR/'tokenizer.json')
        self.probes=None
        if self.config.get('nli'):
            from .intent_probes import Probes
            with zipfile.ZipFile(DIR/'nli.zip') as z:
                self.probes=Probes(z.read('model.onnx'),DIR/'nli-tokenizer.json')
    def assess(self,text):
        cfg=self.config
        w=cfg['semantic_weight']
        x=self.embedder.encode([text]) if w else None
        if self.stacked:
            from .source_review import columns
            vector=columns(x,[text],self.model)
            if self.model.get('lookalikes') is not None:
                vector=np.column_stack((vector,self.model['lookalikes'].decision_function(x)))
            if self.model.get('bags') is not None:
                from .context_windows import aggregate,windows
                pieces=windows(text)
                local=np.vstack((x,self.embedder.encode(pieces,batch_size=1))) if pieces else x
                vector=np.column_stack((vector,aggregate(local,[np.arange(len(local))],self.model['bags'])))
            if self.probes:
                nli=self.probes(text)[None,:]
                vector=np.column_stack((vector,nli,self.model['probes'].decision_function(nli)))
            margin=float(self.model['calibration'].predict_proba(vector)[0,1])
        else:
            lexical=0. if w==1 else float(self.model['lexical'].decision_function([text])[0])
            semantic=float(self.model['semantic'].decision_function(x)[0]) if w else 0.
            margin=w*semantic+(1-w)*lexical
        category=None
        if margin>=cfg['positive']:
            index=int(self.model['subtype'].predict(x if w else self.model['lexical'][0].transform([text]))[0])
            category=self.model['categories'][index]
        return margin<=cfg['negative'],category

@lru_cache(maxsize=1)
def reviewer():
    return Reviewer()

