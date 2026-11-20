"""Run the final cohort with the committed v12 gateway and its original serialized models."""
import io
import subprocess
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def archived_gateway(revision, intent_version):
    """Load the committed runtime before any evaluation module imports app."""
    if "app" in sys.modules:
        raise RuntimeError("Run the baseline in a fresh Python interpreter")
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as checkout:
        archive = subprocess.check_output(
            ["git", "archive", "--format=zip", revision, "app", "security_policy.yaml",
             "experiment/v2/classifier.joblib", "experiment/v3/categories.joblib",
             "experiment/v8/categories.joblib", "experiment/v5/severity.joblib",
             "experiment/v10/jailbreak.joblib", f"experiment/{intent_version}/intent.joblib"], cwd=root)
        with zipfile.ZipFile(io.BytesIO(archive)) as files:
            files.extractall(checkout)
        sys.path.insert(0, checkout)
        try:
            from app import contest
            assert Path(contest.__file__).resolve().parent.parent == Path(checkout).resolve()
            yield
        finally:
            sys.path.remove(checkout)


if __name__ == "__main__":
    with archived_gateway("c9f6a4b", "v12"):
        from experiment.v11.evaluate import run
        from experiment.v13.data import OUT, evaluation
        run("v12", "final", evaluation("final"), OUT)
