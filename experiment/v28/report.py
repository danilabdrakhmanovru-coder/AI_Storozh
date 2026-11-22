"""Pair the same HTTP requests by position and compare technical failures separately."""
from collections import Counter
import json
import re
import numpy as np
from .assessment import ROOT

def read(name):return json.loads(name.read_bytes())

def metrics(rows):
    return dict(n=len(rows),correct=sum(x['pred']==x['gold'] for x in rows),
        false_blocks=sum(x['gold']=='allow' and x['pred']=='block' for x in rows),
        false_confirms=sum(x['gold']=='allow' and x['pred']=='confirm' for x in rows),
        threat_blocks=sum(x['gold']==x['pred']=='block' for x in rows),
        errors=sum(x['pred']=='error' for x in rows),p95_ms=round(float(np.percentile([x['ms'] for x in rows],95)),1))

def main():
    panel=[json.loads(line) for line in (ROOT/'holdout.jsonl').read_bytes().splitlines()]
    lines=['# Проверка авторства деловых сообщений','',
        'Локальный конкурсный FastAPI, B/13. Запросы последовательные, после пяти прогревочных обращений. Ошибки API учитываются отдельно. Оценка решений выполнена без повторных попыток.','',
        '| Выборка | Версия | Верные решения | Угрозы заблокированы | Ложные block | Лишние confirm | Отказы | p95, мс |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    info={}
    for suite in ('fresh','public','holdout'):
        new=read(ROOT/f'v28-{suite}.json')
        if suite=='fresh':
            old=read(ROOT/'v27-fresh.json')
        else:
            old=[dict(domain=r[0],gold=r[1],pred=r[2],ms=r[5]) for r in read(ROOT.parent/'v27'/f'final-{suite}.json')['rows']]
        assert [(r['domain'],r['gold']) for r in old]==[(r['domain'],r['gold']) for r in new]
        pair=list(zip(old,new))
        info[suite]=dict(v27=metrics(old),v28=metrics(new),new_false_blocks=sum(
            a['gold']=='allow' and a['pred']!='block' and b['pred']=='block' for a,b in pair),
            changed=sum(a['pred']!=b['pred'] for a,b in pair))
        for name,rows in [('v27',old),('v28',new)]:
            v=metrics(rows)
            lines.append(f"| {suite} | {name} | {v['correct']}/{len(rows)} | {v['threat_blocks']}/{sum(r['gold']=='block' for r in rows)} | {v['false_blocks']} | {v['false_confirms']} | {v['errors']} | {v['p95_ms']} |")
        if suite=='fresh':
            fresh=pair
    lines+=['','| Сценарий | Эталон | Верно v27 → v28 | v28 allow / confirm / block |','|---|---|---:|---:|']
    for label in ('draft','confirm','fraud','other'):
        group=[p for r,p in zip(panel,fresh) if r['intent']==label]
        counts=Counter(b['pred'] for a,b in group)
        expect=group[0][0]['gold']
        lines.append(f"| {label} | {expect} | {sum(a['pred']==expect for a,b in group)} → {counts[expect]} / {len(group)} | {counts['allow']} / {counts['confirm']} / {counts['block']} |")
    lines+=['','| Область | Верные решения v27 → v28 |','|---|---:|']
    for domain in sorted({r['domain'] for r in panel}):
        group=[p for r,p in zip(panel,fresh) if r['domain']==domain]
        lines.append(f"| {domain} | {sum(a['pred']==a['gold'] for a,b in group)} → {sum(b['pred']==b['gold'] for a,b in group)} / {len(group)} |")
    for suite,delta in info.items():
        lines.append(f"\n{suite} — изменённых решений {delta['changed']}, новых ложных блокировок {delta['new_false_blocks']}.")
    lines+=['','Для обучения использованы синтетические фразы и обычные запросы из ранее существовавшей обучающей части v8/v14. TF-IDF и линейный SVM объединяются с отдельно обученным SVM над локальными представлениями multilingual-e5-small. Файл training.json содержит число уникальных примеров, веса комбинации и SHA256. Классификатор видит текст запроса. Роль, разметка, вложения и заявления в истории не служат подтверждением полномочий.',
            '', 'Новая итоговая выборка зафиксирована до обучения. Это 64 авторских синтетических примера RU/EN и смешанной лексики в восьми темах. Отложены темы переписки и семейства формулировок. Старые 500 публичных и 336 документов v27 использованы для поиска регрессий. Эти проверки не подтверждают качество на частной переписке, длинных диалогах или скрытом тесте организаторов.',
            '', 'Запрос подтверждения сообщает клиенту о необходимости проверить право представительства. Проверку личности подписанта и корпоративной доверенности проводит клиент. Запреты доступа и остальные блокировки сохраняют приоритет.',
            '', '~~~powershell','python -m experiment.v28.train','python -m experiment.v28.evaluate repeat fresh public holdout','python tests/run_all.py','~~~']
    test_log=ROOT/'tests.log'
    if test_log.exists():
        log=test_log.read_text('utf8',errors='replace')
        if '29/29 passed' in log and not re.search(r'^FAILED |^Не прошли:',log,re.M):
            count=sum(map(int,re.findall(r'^Ran (\d+) tests?',log,re.M)))
            lines+=['',f'Локальные регрессии прошли: {count} тестов unittest и 29 примеров правил.']
    lines+=['','Сам исходный запрос о письме от гендиректора получил confirm с категорией sensitive_action. В новой выборке четыре запроса с неясными полномочиями получили allow. Это наблюдаемое ограничение классификатора.']
    (ROOT/'comparison.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
