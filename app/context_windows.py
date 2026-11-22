"""Bounded windows preserve the object of a local request in a longer document."""
import re
import numpy as np

def windows(text):
    parts=re.split(r'(?<=[.!?;:])\s+|[\n\r]+',text)
    packed=[]
    run=''
    for part in parts:
        if run and len(run)+len(part)>260:
            packed.append(run)
            run=''
        if len(part)>330:
            packed.extend([part[i:i+256] for i in range(0,len(part),218)])
        else:run=(run+' '+part).strip()
    if run:packed.append(run)
    unique=list(dict.fromkeys(s.strip() for s in packed if len(s.strip())>=15 and s.strip()!=text.strip()))
    if len(unique)>4:
        unique=[unique[i] for i in np.linspace(0,len(unique)-1,4).round().astype(int)]
    return unique

def aggregate(matrix,groups,heads):
    votes=np.column_stack([h.decision_function(matrix) for h in heads])
    return np.array([np.r_[votes[g].max(axis=0),votes[g].mean(axis=0),votes[g].min(axis=0)] for g in groups])

