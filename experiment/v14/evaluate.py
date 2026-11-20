"""Run frozen packets through the contest entry point or verify them through the batch API."""
import argparse
import json

from experiment.v11.evaluate import ROOT, run
from experiment.v13.data import evaluation as previous
from experiment.v14.data import OUT, evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=("v13", "v14"))
    parser.add_argument("part", choices=("final", "previous", "domains", "public", "api"))
    args = parser.parse_args()
    if args.part == "api":
        from experiment.v11.check_api import check
        check(evaluation(), OUT / f"{args.version}-final.json")
    elif args.part == "domains":
        from experiment.v11.domains import main
        main(args.version, OUT)
    else:
        if args.part == "public":
            rows = [json.loads(s) for s in (ROOT / "artifacts/astra_public/v3-holdout.jsonl").read_bytes().splitlines()]
        else:
            rows = previous("final") if args.part == "previous" else evaluation()
        run(args.version, args.part, rows, OUT)
