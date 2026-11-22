"""Seal the complete CPU runtime and the predeclared evaluation, prior to testing."""
import json
from .data import OUT,ROOT,sha,read,release_digest

def freeze():
    paths=set((ROOT/'app').glob('*.py'))|set((ROOT/'app').glob('*.yaml'))
    paths.update(p for p in (ROOT/'app/weights').rglob('*') if p.is_file())
    runtime_files=['security_policy.yaml','policy.yaml','requirements.txt','requirements-dev.txt',
                   'experiment/v2/classifier.joblib','experiment/v3/categories.joblib','experiment/v8/categories.joblib',
                   'experiment/v5/severity.joblib','experiment/v9/jailbreak.joblib','experiment/v10/jailbreak.joblib',
                   'experiment/v14/intent.joblib','artifacts/astra_public/evaluate.py']
    paths.update(ROOT/p for p in runtime_files)
    paths.update(OUT/name for name in ('panels.json','panels.sha256','data.py','train.py','fit-index.json','human.txt',
                 'float-training.json','evaluate.py','fetch.py','lock.py','float_model.py','quantize.py',
                 'pipeline-training.json','tune_pipeline.py',
                 'short_model.py','recovery.py','v26-runtime.zip'))
    seal={p.relative_to(ROOT).as_posix():release_digest(p) for p in sorted(paths)}
    dst=OUT/'candidate.json'
    if dst.exists():
        if read(dst)!=seal:raise ValueError('Frozen candidate changed')
    else:
        with dst.open('x',encoding='utf8') as f:json.dump(seal,f,indent=1)
    print(len(seal),'files;',sha(dst))

if __name__=='__main__':freeze()
