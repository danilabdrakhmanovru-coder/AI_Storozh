"""Reproduce v13 predictions for both v14 cohorts using the original CPU weights."""
from experiment.v13.baseline import archived_gateway


if __name__ == "__main__":
    with archived_gateway("ceff035", "v13"):
        from experiment.v11.evaluate import run
        from experiment.v14.data import OUT, evaluation
        for cohort in ("final", "confirmation"):
            run("v13", cohort, evaluation(cohort), OUT)
