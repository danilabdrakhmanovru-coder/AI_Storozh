"""Generate totals and per-source counts from the frozen paired run."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    rows = [json.loads(line) for line in (ROOT/"results.jsonl").read_text().splitlines()]
    manifest = json.loads((ROOT/"manifest.json").read_text())
    assert Counter(r["sha"] for r in rows) == Counter(r["sha"] for r in manifest["rows"])
    lines = ["# Независимая проверка на внешних наборах", "",
             "120 русскоязычных переводов из двух публичных наборов. Метки взяты у авторов. "
             "По 30 угроз и 30 обычных запросов из каждого источника. Seed 20261120. "
             "До выбора исключены пустые, длиннее 10 000 символов, точные повторы и совпадения с обучающими и прошлыми тестовыми текстами после удаления крайних пробелов и приведения к нижнему регистру.",
             "", "Модели анализировали текст без фактического выполнения команд. Порог подтверждения 0,35, блокировки 0,75. "
             "Совместный режим выбирает более строгое решение v2 и Qwen. Сбои вынесены отдельно. "
             "Проверки ролей и инструментов сайта в эксперименте не участвовали.", ""]
    names = {"local":"Классификатор v2","qwen":"Qwen 2.5 7B","combined":"Обе модели"}
    for source in ["Все запросы", *manifest["sources"]]:
        subset = [r for r in rows if source=="Все запросы" or r["source"]==source]
        lines += ["## "+source, "", "| Модель | Класс | Пропуск | Подтверждение | Блокировка | Сбой |",
                  "|---|---|---:|---:|---:|---:|"]
        for name, title in names.items():
            for label, group in [(1,"Угроза"),(0,"Обычный")]:
                c = Counter(r[name] for r in subset if r["label"]==label)
                lines.append("| "+" | ".join([title,group,*[str(c[k]) for k in ["allow","confirm","block","error"]]])+" |")
        lines.append("")
    full = json.loads((ROOT/"full-local.json").read_text())
    lines += ["## Дополнительный прогон v2 по двум исходным CSV", "",
              "| Класс | Проверено | Пропуск | Подтверждение | Блокировка |",
              "|---|---:|---:|---:|---:|"]
    for label, title in [("threat","Угроза"),("safe","Обычный")]:
        c = Counter()
        for source_counts in full.values():
            c.update(source_counts[label])
        lines.append(f"| {title} | {sum(c.values())} | {c['allow']} | {c['confirm']} | {c['block']} |")
    lines += ["", "Этот дополнительный прогон охватывает только v2. При сравнении моделей используйте общую выборку из 120 выше.", "",
              "## Что дала вторая модель", ""]
    rescued = sum(r["label"] == 1 and r["qwen"] == "allow" and r["combined"] in {"confirm","block"} for r in rows)
    extra = sum(r["label"] == 0 and r["qwen"] == "allow" and r["combined"] in {"confirm","block"} for r in rows)
    lines.append(f"Добавление v2 к Qwen удержало ещё {rescued} угроз и задержало ещё {extra} обычных запросов, которые Qwen пропускала.")
    lines += ["## Границы интерпретации","",
              "Разметка внешнего корпуса задаёт эталон для этого теста. Доля угроз 50% задана при отборе. "
              "Естественная частота атак в рабочем трафике может быть другой. "
              "Машинный перевод, похожие формулировки и возможное знакомство Qwen с публичными данными ограничивают перенос выводов на реальные запросы. "
              "Проверка на всех возможных запросах по конечной выборке невозможна.", "",
              "## Воспроизведение", "",
              "Файлы и ревизии источников, хеши текстов и модели зафиксированы в manifest.json. "
              "Решения для каждого запроса записаны в results.jsonl. Номера index отсчитываются от нуля после заголовка CSV. "
              "Из корня проекта: `python -m experiment.external.run` и `python -m experiment.external.report`. "
              "Для нового прогона используйте отдельную копию папки без results.jsonl. Существующий прогон продолжается с последней сохранённой записи."]
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

if __name__=="__main__":
    main()
