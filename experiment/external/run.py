"""Frozen stratified sample; no training, tool calls or target-agent dispatch."""
import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path
from app.classifier import classify, MODEL_PATH
from app.model import Ollama
from experiment.compare import decision, load

HERE = Path(__file__).resolve().parent
SOURCES = {
    "benchmark1": "https://huggingface.co/datasets/shalanova/benchmark-1-russian-gt/resolve/d3004d1698dc89eb44abe65264c4f26e824d6a5e/jayavibhav-prompt-injection-safety-russian-gt.csv",
    "benchmark2": "https://huggingface.co/datasets/shalanova/benchmark-2-russian-gt/resolve/25b159cb88242715047afac06adccb8436beccfb/xTRam1safe-guard-prompt-injection-russian-gt.csv",
}
SOURCE_HASHES = {'benchmark1': '50d1bd587f4f18245382f827e5f64e70f75a18574fd22e797ec5f7426104cfae', 'benchmark2': '5f5f4906ab28a5744f3c04fda689b1030b42fecc4ec83fdbf396258ea50cda43'}

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def sample():
    rng = random.Random(20261120)
    seen = {t.strip().casefold() for f in ["train.txt","test.txt","v2/additional.txt","v2/test.txt"] for _,t in load(f)}
    selected = []
    for source, url in SOURCES.items():
        cache = HERE/(source+".csv")
        if cache.exists():
            content = cache.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            cache.write_bytes(content)
        if hashlib.sha256(content).hexdigest() != SOURCE_HASHES[source]:
            raise ValueError("Dataset checksum mismatch: " + source)
        raw = content.decode("utf-8-sig")
        groups = {0:[], 1:[]}
        for i, row in enumerate(csv.DictReader(io.StringIO(raw))):
            text = row["translation"].strip()
            if not text or len(text)>10000 or text.casefold() in seen:
                continue
            seen.add(text.casefold())
            label = int(row["label"])
            groups[label].append(dict(source=source,index=i,label=label,sha=digest(text),text=text))
        for group in groups.values():
            selected.extend(rng.sample(group, 30))
    rng.shuffle(selected)
    return selected

def main():
    judge = Ollama()
    judge.url = "http://127.0.0.1:11434"
    judge.model = "qwen2.5:7b"
    judge.enabled = True
    assert judge.status()["available"]
    rows = sample()
    manifest = {"sources":SOURCES,"seed":20261120,"model_sha256":hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                "thresholds":[.35,.75],"qwen":judge.model,"rows":[{k:v for k,v in r.items() if k!="text"} for r in rows]}
    path = HERE/"manifest.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == manifest
    else:
        path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    out = HERE/"results.jsonl"
    done = {json.loads(line)["sha"] for line in out.read_text().splitlines()} if out.exists() else set()
    rank = ["allow","confirm","block"]
    for i,row in enumerate(rows,1):
        if row["sha"] in done:
            continue
        text = row.pop("text")
        local = classify(text)
        assert local["status"]=="ok"
        q = judge.judge({"text":text})
        if q["status"] != "ok":
            q = judge.judge({"text":text})
        row["local_score"]=local["risk"]
        row["qwen_score"]=q.get("risk")
        row["local"]=decision(local["risk"])
        row["qwen"]=decision(q["risk"]) if q["status"]=="ok" else "error"
        row["combined"]=max(row["local"], row["qwen"], key=rank.index) if q["status"]=="ok" else "error"
        with out.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n")
        if i%10==0:print(i,flush=True)
    print("complete",flush=True)

if __name__=="__main__":
    main()
