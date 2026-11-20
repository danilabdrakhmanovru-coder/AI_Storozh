# Запускает тестовые файлы из подпапок в отдельных процессах, чтобы настройки API не влияли на тесты правил.
import os
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    tests = sorted(Path(__file__).resolve().parent.glob("*/test_*.py"))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    failed = []
    for test in tests:
        print(f"\n>>> {test.parent.name}/{test.name}", flush=True)
        if subprocess.run([sys.executable, str(test)], env=env).returncode:
            failed.append(test.name)
    if failed:
        print("Не прошли: " + ", ".join(failed))
    sys.exit(1 if failed or not tests else 0)
