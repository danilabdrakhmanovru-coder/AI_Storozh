"""Unseen RU/EN packets for paired cross-domain evaluation, with external rows by index."""
from pathlib import Path
from experiment.v11.data import make_row, normal, partition, source

OUT = Path(__file__).resolve().parent
EXTERNAL_JB = """5969 3398 2555 7963 7892 6171 4989 2457 766 6577 1166 4185 2047
573 6054 5243 3962 1759 3471 3332 3083 4945 2490 1408 4731 6911 7475 4389 6254
6362 7796 548 4692 7656 4287 1890 2231 788 3867 8006 7283 7291 622 4565"""


def sector_rows(path):
    result = []
    for line in path.read_text("utf8").splitlines():
        domain, group, *texts = line.split("|")
        for prompt in texts:
            row = make_row(f"{domain}-{len(result)+1:03}", group, prompt)
            row["domain"] = domain
            result.append(row)
    return result


def evaluation():
    result = sector_rows(OUT / "sectors.txt")
    rows = source()
    safe, candidates = partition(rows)
    old = (OUT.parent / "v11/labels.txt").read_text().splitlines()
    excluded = {int(i) for line in old for i in line.split("|")[1].split()}
    attacks = list(map(int, EXTERNAL_JB.split()))
    assert set(attacks) <= set(candidates) - excluded
    for group, ids in [("ordinary", safe[750:950]), ("jailbreak", attacks)]:
        for n, i in enumerate(ids):
            row = make_row(f"external-{i}", group, rows[i]["translation" if n%2 else "text"].strip())
            row["domain"] = "external"
            result.append(row)
    assert len({normal(r["user_prompt"]) for r in result}) == len(result)
    return result
