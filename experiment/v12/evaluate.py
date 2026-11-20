"""Paired gateway decisions. Run the baseline before activating the next artifact."""
import argparse
import json

from experiment.v11.evaluate import ROOT, run
from experiment.v11.data import evaluation as previous
from experiment.v12.data import OUT, evaluation


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("version", choices=("v11", "v12"))
    p.add_argument("sample", choices=("new", "previous", "public", "domains", "api"))
    args = p.parse_args()
    if args.sample == "api":
        from experiment.v11.check_api import check
        check(evaluation(), OUT / f"{args.version}-new.json")
        raise SystemExit
    if args.sample == "domains":
        from experiment.v11.domains import main
        main(args.version, OUT)
        raise SystemExit
    elif args.sample == "public":
        rows = [json.loads(s) for s in (ROOT / "artifacts/astra_public/v3-holdout.jsonl").read_bytes().splitlines()]
    else:
        rows = evaluation() if args.sample == "new" else previous()
    run(args.version, args.sample, rows, OUT)
