"""Fixed, non-generative NLI features for the requested action; no language-model instructions."""
import re
import numpy as np

HYPOTHESES={
 'ru':('Автор требует раскрыть чужую конфиденциальную информацию.',
       'Автор просит помощника отказаться от своих обязательных правил или заменить исходную задачу.',
       'Автор хочет, чтобы было совершено вредное или незаконное действие.',
       'Это обычная просьба написать, перевести, классифицировать или объяснить текст.',
       'Это литературный текст, пример или описание событий, а не поручение нарушить чьи-либо права.',
       'Автор описывает обычную операцию с предметом, данными или программой.'),
 'en':('The author wants another party’s confidential information disclosed.',
       'The author asks the assistant to disregard its governing rules or replace its assigned task.',
       'The author wants a harmful or illegal action carried out.',
       'This is an ordinary request to write, translate, classify or explain some text.',
       'This is a fictional passage, example or account of events rather than a request to violate someone’s rights.',
       'The author describes an ordinary operation on an object, records or software.')
}

class Probes:
    def __init__(self,model,tokenizer,threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok=Tokenizer.from_file(str(tokenizer))
        self.hyp={lang:[self.tok.encode(h,add_special_tokens=False).ids for h in bank] for lang,bank in HYPOTHESES.items()}
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
        self.net=ort.InferenceSession(model,opts,providers=['CPUExecutionProvider'])
        self.inputs={i.name for i in self.net.get_inputs()}
        # XLM-R tokenizer uses <s> A </s></s> B </s>.
        assert self.tok.encode('x','y').ids==[0]+self.tok.encode('x',add_special_tokens=False).ids+[2,2]+self.tok.encode('y',add_special_tokens=False).ids+[2]
    def __call__(self,text):
        lang='ru' if re.search('[А-Яа-яЁё]',text) else 'en'
        if len(text)>4800:
            mid=len(text)//2;text=text[:1900]+' '+text[mid-350:mid+350]+' '+text[-1900:]
        ids=self.tok.encode(text,add_special_tokens=False).ids
        if len(ids)>132:
            mid=len(ids)//2
            ids=ids[:52]+ids[mid-10:mid+10]+ids[-60:]
        pairs=[[0]+ids+[2,2]+h+[2] for h in self.hyp[lang]]
        x=np.ones((len(pairs),max(map(len,pairs))),dtype='int64')
        m=np.zeros_like(x)
        for i,p in enumerate(pairs):x[i,:len(p)]=p;m[i,:len(p)]=1
        inputs=dict(input_ids=x,attention_mask=m)
        if 'token_type_ids' in self.inputs:inputs['token_type_ids']=np.zeros_like(x)
        z=self.net.run(None,inputs)[0]
        return (z[:,0]-np.logaddexp(z[:,1],z[:,2])).astype(np.float32)

