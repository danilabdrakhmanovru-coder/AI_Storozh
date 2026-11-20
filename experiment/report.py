"""Build a readable report from a completed comparison."""
import json
from statistics import median
from experiment.compare import ROOT

def main():
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    rows = results["rows"]
    report = ["# Результаты пилотного сравнения", "",
              "40 обучающих фраз, 16 отдельных тестовых фраз. Все тексты синтетические.", "",
              "| Модель | Тип запроса | Разрешено | На подтверждение | Блокировано | Сбой |",
              "|---|---|---:|---:|---:|---:|"]
    for model, name in [("local", "Свой классификатор"), ("qwen", results["model"])]:
        for label, group in [("threat", "Угрозы"), ("safe", "Обычные")]:
            counts = results["counts"][model][label]
            values = [str(counts.get(d, 0)) for d in ["allow","confirm","block","error"]]
            report.append("| " + " | ".join([name, group, *values]) + " |")
    qwen_ms = [r["qwen_details"]["latency_ms"] for r in rows if r["qwen"] != "error"]
    report += ["", f"Медиана времени своего классификатора — {median(r['local_ms'] for r in rows):.2f} мс."]
    if qwen_ms:
        report.append(f"Медиана времени успешного запроса к Qwen — {median(qwen_ms)/1000:.2f} с.")
    report += ["", "| Ожидаемый класс | Запрос | Свой классификатор | Qwen |", "|---|---|---|---|"]
    for r in rows:
        report.append(f"| {r['label']} | {r['text']} | {r['local']} | {r['qwen']} |")
    (ROOT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
