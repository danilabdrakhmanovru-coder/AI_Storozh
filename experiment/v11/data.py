"""Fixed bilingual samples from the archived public corpus, with source-level separation."""
import csv
import hashlib
import io
import json
import random
import re
from pathlib import Path

from experiment.v10.benchmark import HERE, SOURCE_HASHES, ROLE_CASES, ORDINARY_CASES, make_row

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "v10"


def source():
    raw = (HERE / "benchmark2.csv").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_HASHES["benchmark2"]
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def normal(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def partition(rows):
    seen = {r["index"] for path in (HERE / "manifest.json", OUT.parent / "v9/external-control.json")
            for r in json.loads(path.read_bytes())["rows"] if r["source"] == "benchmark2"}
    seen.update(map(int, (ROLE_CASES + " " + ORDINARY_CASES).split()))
    for line in (OLD / "final-indices.txt").read_text().splitlines():
        seen.update(map(int, line.split("|")[1].split()))
    used = {normal(rows[i][field]) for i in seen for field in ("text", "translation")}
    ids = list(range(len(rows)))
    random.Random(11037).shuffle(ids)
    safe, attacks = [], []
    for i in ids:
        row = rows[i]
        keys = {normal(row[f]) for f in ("text", "translation")}
        if i in seen or used & keys or "" in keys or max(map(len, keys)) > 10000:
            continue
        used.update(keys)
        (safe if row["label"] == "0" else attacks).append(i)
    return safe, attacks


def split(rows):
    safe, attacks = partition(rows)
    return safe[:500], safe[500:750], attacks


def evaluation():
    rows = source()
    _, safe, _ = split(rows)
    groups = {"ordinary": safe}
    for line in (OUT / "labels.txt").read_text().splitlines():
        name, indices = line.split("|")
        groups[name] = list(map(int, indices.split()))
    result = []
    for group, indices in groups.items():
        for j, index in enumerate(indices):
            field = "text" if j % 2 == 0 else "translation"
            result.append(make_row(f"ext-{index}", group, rows[index][field].strip()))
    for j, line in enumerate((OUT / "control.txt").read_text("utf8").splitlines()):
        group, text = line.split("|", 1)
        result.append(make_row(f"control-{j+1}", group, text))
    assert len({r["id"] for r in result}) == len(result)
    return result
