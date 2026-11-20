"""Render the completed v2 benchmark in Russian."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
NAMES = {"v1":"Свой v1", "v2":"Свой v2", "qwen":"Qwen 2.5 7B"}
LABELS = {"safe":"Обычный", "threat":"Угроза"}
DECISIONS = {"allow":"Пропуск", "confirm":"Подтверждение", "block":"Блокировка", "error":"Сбой"}

def main():
    data = json.loads((OUT/"results.json").read_text(encoding="utf-8"))
    lines = ["# Сравнение на новых запросах", "",
             "На 24 новых русскоязычных фразах сравнивались две версии своего классификатора и локальная Qwen. "
             "Тест содержит 12 обычных запросов и 12 угроз. Тексты и метки составлены автором эксперимента до запуска. "
             "Первая версия обучена на 40 фразах, вторая на 120 (68 обычных, 52 угрозы). "
             "Архитектура, настройки обучения и пороги решений сохранены. Прежний тест в обучение не включался. "
             "Результаты отражают работу текстовых моделей без правил серверного шлюза.", "",
             "| Модель | Класс | Пропуск | Подтверждение | Блокировка | Сбой |", "|---|---|---:|---:|---:|---:|"]
    for model, groups in data["counts"].items():
        for label, counts in groups.items():
            values = [str(counts.get(k, 0)) for k in DECISIONS]
            lines.append("| " + " | ".join([NAMES[model],LABELS[label],*values]) + " |")
    regressions = [r for r in data["rows"] if r["label"]=="threat" and r["v1"]!="allow" and r["v2"]=="allow"]
    lines += ["", f"Угроз, впервые пропущенных после переобучения на этом наборе: {len(regressions)}.",
              "", "| Запрос | Метка | v1 | v2 | Qwen |", "|---|---|---|---|---|"]
    for row in data["rows"]:
        lines.append("| " + " | ".join([row["text"],LABELS[row["label"]],*[DECISIONS[row[m]] for m in NAMES]]) + " |")
    lines += ["","## Воспроизведение","",
              "Из корня проекта с установленными зависимостями эксперимента и работающей Ollama выполните "
              "`python -B -m experiment.v2.run`, затем `python -B -m experiment.v2.report`. "
              "Артефакт обученной модели сохраняется в experiment/v2/classifier.joblib. "
              "В results.json доступны индивидуальные оценки всех трёх моделей и пояснения Qwen."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

if __name__=="__main__":
    main()
