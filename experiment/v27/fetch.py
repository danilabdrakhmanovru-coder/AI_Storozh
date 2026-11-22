"""Restore pinned external evaluation inputs for research; inference never imports this module."""
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
from .data import OUT,RAW,sha,read

SUITE='https://huggingface.co/datasets/Horizon-Labs/prompt-injection-eval-suite/resolve/d6dd8d499d0087661afd6cd7ecdfcf94a5af1c05/data/'
RAG='https://huggingface.co/datasets/heisenberg-88/indirectrag-bench/resolve/93254b5e80ba1345c2a25aec2b44fcf479bf52f5/indirectrag_bench.jsonl'
FILES=[
    ('v24-documents.parquet',SUITE+'synthetic_docs_test.parquet','bc7d4fd5430b3424f1b1271e57ca5c00da08d470a9eaa4399e04df67adafc23f'),
    ('v22-piarena.parquet',SUITE+'piarena.parquet','0bbd59a14b622e0ed8630cac365c136537c2ab2b31e4b3fd9587b5edc0b88b9a'),
    ('v26-horizon.parquet',SUITE+'synthetic_v2_test.parquet','e8d4b34e5ff318e4a33c66f460f02bd070ecb747fc718ee941ff83e645febfef'),
    ('v26-retrieved.jsonl',RAG,'46bf29947e2b201696c559e9867d281720530488cc787d0ec576a7bf5d82b745')]

def prepare():
    import pyarrow.parquet as pq
    RAW.mkdir(exist_ok=True)
    for name,url,digest in FILES:
        p=RAW/name
        if not p.exists():
            with tempfile.TemporaryDirectory(dir=RAW) as tmp:
                dst=Path(tmp)/name
                with urllib.request.urlopen(url,timeout=120) as response,dst.open('wb') as f:
                    shutil.copyfileobj(response,f)
                if sha(dst)!=digest:raise ValueError(name)
                dst.replace(p)
        if sha(p)!=digest:raise ValueError(name)
    def parquet(name):
        return pq.read_table(RAW/name).to_pylist()
    def write(name,rows):
        data=json.dumps(rows,ensure_ascii=False).encode('utf8')
        import hashlib
        if hashlib.sha256(data).hexdigest()!=read(OUT/'panels.json')['hashes'][name]:raise ValueError(name)
        p=RAW/name
        if not p.exists():p.write_bytes(data)
        elif sha(p)!=read(OUT/'panels.json')['hashes'][name]:raise ValueError(name)
    for source,target in [('v24-documents.parquet','v24-documents.json'),('v22-piarena.parquet','v23-piarena.json')]:
        write(target,[[r['text'],int(r['label'])] for r in parquet(source)])
    rag=[json.loads(line) for line in (RAW/'v26-retrieved.jsonl').read_text('utf8').splitlines()]
    write('v26-external.json',dict(
        indirect_v2=[[r['text'],int(r['label'])] for r in parquet('v26-horizon.parquet') if r['kind']=='indirect'],
        retrieved=[['\n\n'.join(r['retrieved_passages']),int(r['injected_passage_index']>=0)] for r in rag]))
    print('Verified external corpora')

if __name__=='__main__':prepare()
