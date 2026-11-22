"""Train a CPU correspondence head. Development and final document panels use separate domains and families."""
from collections import Counter
from itertools import product
from pathlib import Path
import hashlib
import json
import random

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion,make_pipeline
from sklearn.svm import LinearSVC

from app.semantic_review import Reviewer

ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT.parents[1]/'app/weights/correspondence.joblib'
SEED=2813
SUBJECTS={
'ru':[
('директор','директора','контракт','офис партнёра'),
('редактор','редактора','тираж','редакцию'),
('врач','врача','график приёма','клинику'),
('руководитель','руководителя','инвентаризация','склад'),
('мастер','мастера','ремонт','мастерскую'),
('заведующая','заведующей','расписание занятий','детский сад'),
('делегат выставки','делегата выставки','стенд','секретариат'),
('Ирина','Ирины','встреча','клуб'),('партнёр','партнёра','поездка','бюро'),
('председатель комиссии','председателя комиссии','слушания','управу')],
'en':[
('director','director','the agreement','the partner office'),
('editor','editor','the print run','the publisher'),
('physician','physician','consultations','the clinic'),
('manager','manager','the inventory','the warehouse'),
('foreman','foreman','the repair','the workshop'),
('supervisor','supervisor','the timetable','the school'),
('delegate','delegate','the stall','the fair'),('Robin','Robin','a meeting','the club'),
('associate','associate','the trip','the bureau'),('commissioner','commissioner','the hearing','the council')]}

def labeled(path):
    return [line.split('|',1) for line in path.read_text('utf8').splitlines() if line and not line.startswith('#')]

def validation():
    return [(t,{'other':'allow','draft':'allow','confirm':'confirm','fraud':'block'}[label])
            for label,t in labeled(ROOT/'dev.txt')]

def rows():
    rng=random.Random(SEED)
    pairs={}
    for line in (ROOT/'training.txt').read_text('utf8').splitlines():
        if not line or line.startswith('#'):continue
        if line.startswith('['):
            label,language=line[1:-1].split()
        else:
            for n,p,s,r in rng.sample(SUBJECTS[language],4):
                pairs[line.format(n=n,p=p,s=s,r=r)]=label
    for version in ('v8','v14'):
        pool=[t for line in (ROOT.parent/version/'training.txt').read_text('utf8').splitlines()
              if line.startswith('none|') for t in line.split('|')[1:] if t.strip()]
        for t in rng.sample(pool,min(180,len(pool))):
            pairs.setdefault(t,'other')
    return list(pairs.items())

def main():
    from app.correspondence import classify_scores
    samples=rows()
    texts,y=zip(*samples)
    dev_y,dev_text=zip(*labeled(ROOT/'dev.txt'))
    assert not set(texts)&set(dev_text)
    vec=FeatureUnion([
        ('words',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
        ('chars',TfidfVectorizer(analyzer='char',ngram_range=(3,5),sublinear_tf=True,min_df=2))])
    lexical=make_pipeline(vec,LinearSVC(C=1.25,class_weight='balanced',random_state=SEED))
    lexical.fit(texts,y)
    enc=Reviewer().embedder
    train_x=enc.encode(texts,batch_size=4)
    semantic=LinearSVC(C=6,class_weight='balanced',random_state=SEED).fit(train_x,y)
    names=lexical.classes_.tolist()
    assert names==semantic.classes_.tolist()
    a=lexical.decision_function(dev_text)
    # Quantized CPU embeddings depend slightly on the padded batch. Calibrate
    # with the same single-prompt geometry used by the service.
    b=semantic.decision_function(enc.encode(dev_text,batch_size=1))
    best=None
    for weight,min_score,fast,margin in product((.25,.45,.65),(-.45,-.2,0.),(.75,1.1),(.15,.08)):
        opts=dict(semantic_weight=weight,min_attribution=min_score,fast_other=fast,fraud_margin=margin)
        pred=[classify_scores(names,left,right,opts) for left,right in zip(a,b)]
        loss=sum(8*(g=='fraud' and p not in ('fraud','confirm'))+4*(g=='fraud' and p=='confirm')+
                 5*(g in ('other','draft') and p=='fraud')+2*(g in ('other','draft') and p=='confirm')+
                 3*(g=='confirm' and p!='confirm') for g,p in zip(dev_y,pred))
        if best is None or loss<best[0]:best=(loss,opts,pred)
    _,config,pred=best
    joblib.dump(dict(lexical=lexical,semantic=semantic,classes=names,config=config),WEIGHTS,compress=3)
    info=dict(training=dict(Counter(y)),development=dict(Counter(f'{g}/{p}' for g,p in zip(dev_y,pred))),
              config=config,model_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
              training_sha256=hashlib.sha256(json.dumps(samples,ensure_ascii=False).encode()).hexdigest())
    (ROOT/'training.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    print(info,flush=True)

if __name__=='__main__':main()
