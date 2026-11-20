"""Frozen bilingual sector scenarios and archived external ids, sampled before fitting."""
from pathlib import Path

from experiment.v11.data import make_row, normal, partition, source
from experiment.v12.data import EXTERNAL_JB, sector_rows

OUT = Path(__file__).resolve().parent
JAILBREAK = """7242 2867 54 5431 7707 7226 4864 7153 4784 2423 3245 1463 214 7842
4758 2354 5057 7485 3313 6757 5126 832 1997 13 2688 2033 2946 1424"""


FINAL_JB = """4125 7064 2292 219 1105 1010 7731 5354 3002 7012 3784 392 5063 4263 6050"""


def evaluation(part="new"):
    final = part == "final"
    rows = sector_rows(OUT / ("final-sectors.txt" if final else "sectors.txt"))
    for n, row in enumerate(rows):
        prompt = row["user_prompt"]
        if n % 6 == 1:
            row["context"]["history"] = [{"role": "user", "content": "Hello." if prompt.isascii() else "Здравствуйте."}]
        if n % 6 == 3:
            row["context"]["history"] = [{"role": "user", "content": prompt}]
            row["user_prompt"] = "Please continue." if prompt.isascii() else "Продолжи."
    corpus = source()
    safe, attacks = partition(corpus)
    old = {int(i) for line in (OUT.parent / "v11/labels.txt").read_text().splitlines()
           for i in line.split("|")[1].split()}
    old.update(map(int, EXTERNAL_JB.split()))
    if final:
        old.update(map(int, JAILBREAK.split()))
    jb = list(map(int, (FINAL_JB if final else JAILBREAK).split()))
    assert set(jb) <= set(attacks) - old
    for group, ids in [("ordinary", safe[2350:2600] if final else safe[1500:1750]), ("jailbreak", jb)]:
        for n, i in enumerate(ids):
            prompt = corpus[i]["translation" if n % 2 else "text"].strip()
            row = make_row(f"external-{i}", group, prompt)
            row["domain"] = "external"
            rows.append(row)
    return rows


def source_texts(rows):
    return {normal(q) for row in rows
            for q in [row["user_prompt"], *(m["content"] for m in row["context"].get("history", []))]
            if normal(q) not in {"hello", "здравствуйте", "продолжи", "please continue"}}
