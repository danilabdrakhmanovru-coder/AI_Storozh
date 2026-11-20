"""Paired evaluation packets fixed before training or inspecting model predictions."""
from pathlib import Path

from experiment.v11.data import make_row, normal, partition, source
from experiment.v12.data import EXTERNAL_JB, sector_rows
from experiment.v13.data import JAILBREAK, FINAL_JB

OUT = Path(__file__).resolve().parent
PUBLIC_JB = """7928 5996 8010 6626 5045 4094 4099 1079 6283 468 5769 7861 7865 6389
2002 683 113 722 5573 5721 2113 2478 5494 1901"""


def evaluation():
    rows = sector_rows(OUT / "sectors.txt")
    for index, row in enumerate(rows):
        text = row["user_prompt"]
        if index % 5 == 1:
            row["context"]["history"] = [{"role": "user", "content": "Hello." if text.isascii() else "Здравствуйте."}]
        elif index % 5 == 3:
            row["context"]["history"] = [{"role": "user", "content": text}]
            row["user_prompt"] = "Please continue." if text.isascii() else "Продолжи."
    corpus = source()
    safe, attacks = partition(corpus)
    prior = {int(i) for line in (OUT.parent / "v11/labels.txt").read_text().splitlines()
             for i in line.split("|")[1].split()}
    prior.update(map(int, (EXTERNAL_JB + " " + JAILBREAK + " " + FINAL_JB).split()))
    jb = list(map(int, PUBLIC_JB.split()))
    assert set(jb) <= set(attacks) - prior
    for group, ids in [("ordinary", safe[2600:3000]), ("jailbreak", jb)]:
        for n, i in enumerate(ids):
            row = make_row(f"external-{i}", group, corpus[i]["translation" if n % 2 else "text"].strip())
            row["domain"] = "external"
            rows.append(row)
    originals = [r["context"]["history"][-1]["content"] if normal(r["user_prompt"]) in {"продолжи", "please continue"}
                 else r["user_prompt"] for r in rows]
    assert len(set(map(normal, originals))) == len(rows)
    return rows
