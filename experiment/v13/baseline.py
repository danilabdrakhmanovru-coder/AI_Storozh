"""Run the final cohort with the committed v12 gateway and its original serialized models."""
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as checkout:
        archive = subprocess.check_output(
            ["git", "archive", "--format=zip", "c9f6a4b", "app", "security_policy.yaml",
             "experiment/v2/classifier.joblib", "experiment/v3/categories.joblib",
             "experiment/v8/categories.joblib", "experiment/v5/severity.joblib",
             "experiment/v10/jailbreak.joblib", "experiment/v12/intent.joblib"], cwd=root)
        with zipfile.ZipFile(io.BytesIO(archive)) as files:
            files.extractall(checkout)
        sys.path.insert(0, checkout)
        from experiment.v11.evaluate import run
        from experiment.v13.data import OUT, evaluation
        from app import contest
        assert Path(contest.__file__).resolve().parent.parent == Path(checkout).resolve()
        run("v12", "final", evaluation("final"), OUT)
