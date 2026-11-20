"""ИИ-сторож: API and local operator console. Run with uvicorn app.main:app."""
import logging
import yaml
from .contest import ContestRequest, inspect_request
import csv
import hashlib
import html
import io
import json
import os
import secrets
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse
from fastapi import FastAPI, Depends, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from .guard import Policy, check, _level, RECO, COMPILED
from .classifier import classify, evaluate_stage
from .model import Ollama
from .storage import Store
from .cases import CASES

ROOT = Path(__file__).resolve().parent.parent
policy = Policy(os.getenv('STOROZH_POLICY', str(ROOT/'policy.yaml')))
store = Store(os.getenv('STOROZH_DB', str(ROOT/'audit.db')))
model = Ollama()
app = FastAPI(title='ИИ-сторож · защитный шлюз', version='0.3.0')
app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')

GROUPS = [
 ('instructions','Целостность инструкций','Jailbreak, подмена и извлечение системного промпта', {'prompt_injection','jailbreak','prompt_extraction','instruction_tampering','indirect_injection','evasion'}),
 ('data','Защита данных','Секреты, чужие данные и попытки утечки', {'sensitive_access','unauthorized_data','data_exfiltration','secret_leak'}),
 ('actions','Безопасность действий','Опасные команды и аргументы инструментов', {'dangerous_command','tool_abuse'}),
 ('access','Права и политики','Роль пользователя и доступные инструменты', {'privilege_escalation','tool_abuse','unauthorized_data'})]
LABELS = {'prompt_injection':'Попытка отменить инструкции','jailbreak':'Обход ограничений','prompt_extraction':'Извлечение системных инструкций',
 'instruction_tampering':'Подмена инструкций','indirect_injection':'Инструкция в недоверенном источнике','evasion':'Сокрытие инструкций',
 'sensitive_access':'Доступ к защищённым ресурсам','unauthorized_data':'Чужие данные','data_exfiltration':'Возможная утечка данных',
 'secret_leak':'Секрет в запросе','dangerous_command':'Опасная команда','tool_abuse':'Использование инструментов',
 'privilege_escalation':'Попытка повысить полномочия','classifier_risk':'Риск по оценке классификатора v2','model_risk':'Риск по оценке Qwen','model_error':'Сбой модели','internal_error':'Сбой правил'}

def version():
    return hashlib.sha256(json.dumps(policy.data, sort_keys=True).encode()).hexdigest()[:16]

def identity(request: Request, x_api_key: Optional[str] = Header(None)):
    admin_key=os.getenv('STOROZH_API_KEY','')
    clients=json.loads(os.getenv('STOROZH_CLIENTS_JSON','{}'))
    # A same-origin local console is the explicit demonstration boundary.
    origin=request.headers.get('origin')
    if origin and urlparse(origin).netloc != request.headers.get('host'):
        raise HTTPException(403,'Запрос с другого сайта запрещён')
    if admin_key or clients:
        if admin_key and secrets.compare_digest(x_api_key or '',admin_key):
            return {'user_id':'operator','role':'admin','admin':True,'demo':False}
        for key, value in clients.items():
            if secrets.compare_digest(x_api_key or '',key):
                return {**value,'admin':False,'demo':False}
        raise HTTPException(401,'Требуется действующий X-API-Key')
    host=urlparse('http://'+request.headers.get('host','')).hostname
    if request.client.host not in {'127.0.0.1','::1','testclient'} or host not in {'127.0.0.1','localhost','::1','testserver'}:
        raise HTTPException(403,'Без ключей доступна только локальная демонстрация')
    return {'user_id':'demo','role':'guest','admin':True,'demo':True}

def operator(who=Depends(identity)):
    if not who['admin']: raise HTTPException(403,'Нужны права оператора')
    return who

class ToolCall(BaseModel):
    name: str = Field(min_length=1,max_length=80)
    args: dict = Field(default_factory=dict)

class CheckRequest(BaseModel):
    text: str = Field(min_length=1,max_length=10000)
    user_id: str = Field(default='demo',min_length=1,max_length=100)
    role: Optional[str] = Field(default=None,max_length=80)
    source: Literal['user','document','tool_output','web'] = 'user'
    tools: list[ToolCall] = Field(default_factory=list,max_length=10)
    @model_validator(mode='after')
    def bounded(self):
        if not self.text.strip(): raise ValueError('Введите непустой запрос')
        if len(self.model_dump_json()) > 12000: raise ValueError('Запрос с аргументами превышает 12 000 символов')
        return self

def run_check(req, who):
    start=time.perf_counter()
    payload=req.model_dump()
    if not who['admin']:
        payload.update(user_id=who['user_id'],role=who['role'])
    payload['role']=payload['role'] or policy.data['default_role']
    # Snapshot ensures rules, thresholds and recorded policy version agree during edits.
    snapshot=type('PolicySnapshot',(),{'data':policy.data})()
    pv=hashlib.sha256(json.dumps(snapshot.data,sort_keys=True).encode()).hexdigest()[:16]
    r=check(payload['text'],snapshot,payload['role'],payload['user_id'],payload['tools'],source=payload['source'],recent_blocks=store.blocks(payload['user_id']))
    rules_decision=r.decision
    rules = {"decision": r.decision, "risk": r.risk, "reasons": list(r.reasons)}
    thresholds = snapshot.data["thresholds"]
    local = evaluate_stage(classify(payload["text"]), thresholds)
    j = evaluate_stage(model.judge(payload), thresholds)
    rank = {"allow": 0, "confirm": 1, "block": 2}
    for stage, category in [(local, "classifier_risk"), (j, "model_risk")]:
        r.decision = max(r.decision, stage["decision"], key=rank.get)
        r.risk = max(r.risk, stage.get("risk", thresholds["confirm"]))
        if stage["decision"] != "allow":
            r.categories.append(category if stage["status"] == "ok" else "internal_error")
            r.reasons.append(stage["model"] + ": " + stage["reason"])
    r.level=_level(r.risk)
    r.recommendation=RECO[r.decision]
    r.latency_ms=round((time.perf_counter()-start)*1000,1)
    out=asdict(r)
    out['request_id']=uuid.uuid4().hex
    out['classifier']=local
    out['rules']=rules
    out['model']=j
    out['rules_decision']=rules_decision
    out['policy_version']=pv
    out['identity']={'user_id':payload['user_id'],'role':payload['role']}
    out['checks']=[]
    for key,title,desc,cats in GROUPS:
        found=[c for c in r.categories if c in cats]
        details=[LABELS.get(c,c) for c in found]
        if key=='access':
            details += [reason for reason in r.reasons if 'роль' in reason or 'роли' in reason or 'политике' in reason or 'пользователя' in reason]
        out['checks'].append({'id':key,'title':title,'description':desc,'status':'attention' if details else 'clear',
                              'details':list(dict.fromkeys(details)) or ['Нарушений по правилам не обнаружено']})
    out['summary']={'allow':'Запрос прошёл включённые проверки. Его можно передать целевому агенту.',
                    'block':'Передача остановлена: обнаружены признаки атаки или нарушение политики.',
                    'confirm':'Запрос удержан. Для продолжения требуется решение оператора.'}[r.decision]
    stages = {"правила": rules, "классификатор v2": local, "Qwen": j}
    decisive = [name for name, stage in stages.items() if stage["decision"] == r.decision]
    out["summary"] += " Основание решения: " + ", ".join(decisive) + "."
    incomplete = local["status"] != "ok" or j["status"] != "ok"
    out["can_review"] = r.decision == "confirm" and not incomplete and "internal_error" not in r.categories
    if incomplete or "internal_error" in r.categories:
        out["summary"] = ("Передача запрещена." if r.decision == "block" else "Передача удержана.") + " Проверка не завершена. Повторите запрос после восстановления всех компонентов."
    out['can_forward']=r.decision=='allow'
    if r.decision=='confirm':out['confirm_id']=out['request_id']
    store.add(payload,out,pv)
    return out

@app.get('/health')
def health():return {'status':'ok','version':'0.3.0'}

@app.get('/config')
def config(who=Depends(identity)):
    return {'demo':who['demo'],'operator':who['admin'],'identity':who,'model':model.status(),
            'policy':policy.data,'groups':[{'id':k,'title':t,'description':d} for k,t,d,_ in GROUPS], 'labels':LABELS}

@app.post('/check')
def do_check(req:CheckRequest,who=Depends(identity)):return run_check(req,who)

@app.post('/check/batch')
def batch(reqs:list[CheckRequest],who=Depends(identity)):
    if not 1<=len(reqs)<=20:raise HTTPException(422,'От 1 до 20 запросов в пакете')
    return [run_check(r,who) for r in reqs]

class Review(BaseModel):
    approve:bool

@app.post('/confirm/{rid}')
def confirm(rid:str,body:Review,who=Depends(operator)):
    try:return store.review_request(rid,body.approve,who['user_id'])
    except ValueError as e:raise HTTPException(409,str(e))

@app.post('/dispatch/{rid}')
def dispatch(rid:str,who=Depends(identity)):
    try:r=store.claim(rid,who['user_id'],who['admin'],version())
    except ValueError as e:raise HTTPException(409,str(e))
    try:
        answer=model.target(r['text'])
        store.dispatched(rid,answer)
        return {'request_id':rid,'answer':answer,'state':'delivered'}
    except Exception:
        store.dispatched(rid)
        raise HTTPException(502,'Агент не ответил. Автоматической повторной отправки нет; для повторения проверьте запрос заново.')

@app.get('/requests/{rid}')
def detail(rid:str,who=Depends(identity)):
    r=store.get(rid)
    if not r or (not who['admin'] and r['user_id']!=who['user_id']):raise HTTPException(404,'Запрос не найден')
    return r

class Feedback(BaseModel):
    label:Literal['correct','false_positive','missed_attack']
    note:str=Field(default='',max_length=1000)

@app.put('/requests/{rid}/feedback')
def feedback(rid:str,body:Feedback,who=Depends(operator)):
    try:store.feedback(rid,body.label,body.note)
    except ValueError as e:raise HTTPException(404,str(e))
    return {'ok':True}

@app.get('/policy')
def get_policy(who=Depends(identity)):return policy.data

@app.put('/policy')
def put_policy(data:dict,who=Depends(operator)):
    try:policy.save(data)
    except (ValueError,TypeError,AttributeError) as e:raise HTTPException(422,str(e))
    return {'ok':True,'version':version()}

def bounds(start,end):
    if end<=start:raise HTTPException(422,'Конец периода должен быть позже начала')

@app.get('/logs')
def logs(start:float=Query(0,ge=0,allow_inf_nan=False),end:float=Query(4102444800,ge=0,allow_inf_nan=False),
         decision:Optional[Literal['allow','confirm','block']]=None,n:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0),who=Depends(operator)):
    bounds(start,end)
    return store.rows(start,end,decision,n,offset)

@app.get('/reports')
@app.get('/stats')
def report(start:float=Query(0,ge=0,allow_inf_nan=False),end:float=Query(4102444800,ge=0,allow_inf_nan=False),who=Depends(operator)):
    bounds(start,end)
    return store.report(start,end)

@app.get('/reports/export')
def export(start:float=Query(0,ge=0,allow_inf_nan=False),end:float=Query(4102444800,ge=0,allow_inf_nan=False),
           format:Literal['html','csv']='html',who=Depends(operator)):
    bounds(start,end)
    rep=store.report(start,end)
    if format=='csv':
        buffer=io.StringIO(newline='');writer=csv.writer(buffer)
        writer.writerow(['id','time_utc','user','role','decision','risk','summary','categories','model_status','state','feedback'])
        def safe(value):
            s=str(value)
            return "'"+s if s.lstrip().startswith(('=','+','-','@')) or s.startswith(('\t','\r','\n')) else s
        for r in store.rows(start,end):
            writer.writerow([safe(x) for x in [r['id'],time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(r['ts'])),r['user_id'],r['role'],r['decision'],r['result']['risk'],r['result']['summary'],', '.join(r['result']['categories']),r['result']['model']['status'],r['state'],r['feedback']['label'] if r['feedback'] else '']])
        return Response('\ufeff'+buffer.getvalue(),media_type='text/csv; charset=utf-8',headers={'Content-Disposition':'attachment; filename="storozh-report.csv"'})
    e=html.escape
    cats=''.join(f'<tr><td>{e(LABELS.get(k,k))}</td><td>{v}</td></tr>' for k,v in rep['categories'].items()) or '<tr><td>Срабатываний нет</td><td>0</td></tr>'
    errors=''.join(f'<li>{e(k)}: {v}</li>' for k,v in rep['feedback'].items()) or '<li>Оператор ещё не оценивал решения.</li>'
    date=lambda t:time.strftime('%d.%m.%Y %H:%M UTC',time.gmtime(t))
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Отчёт ИИ-сторожа</title><style>body{{font:16px/1.6 system-ui;max-width:900px;margin:60px auto;padding:24px;color:#172d2a}}h1{{font-size:38px}}small{{color:#52645f}}table{{width:100%;border-collapse:collapse}}td{{padding:12px;border-bottom:1px solid #ddd}}.box{{background:#eef5f2;padding:24px;border-radius:16px}}@media print{{button{{display:none}}body{{margin:0}}}}</style><small>ИИ-СТОРОЖ / ОТЧЁТ БЕЗОПАСНОСТИ</small><h1>Контроль запросов к ИИ</h1><p>{date(start)} — {date(end)}</p><div class="box">{e(rep['summary'])}</div><h2>Частые срабатывания</h2><table>{cats}</table><h2>Качество и стабильность</h2><p>Сбоев проверки: {rep['technical_errors']}. p95: {rep['p95_ms']} мс. Подтверждений обработано: {rep['reviewed']}. Передано агенту: {rep['delivered']}.</p><ul>{errors}</ul><p>{e(rep['quality_note'])}</p><h2>Рекомендации</h2><p>{'Проверьте доступность Ollama и повторите незавершённые проверки.' if rep['technical_errors'] else 'Отмечайте неверные решения в журнале и расширяйте независимый тестовый набор.'}</p><small>Отчёт включает проверки выбранного периода; статусы подтверждения и отзывы приведены на момент выгрузки. Тексты запросов не включены. Несколько категорий могут относиться к одному запросу.</small><p><button onclick="window.print()">Печать / сохранить PDF</button></p></html>'''
    return Response(page,media_type='text/html',headers={'Content-Disposition':'attachment; filename="storozh-report.html"'})

@app.get('/selftest')
def selftest(who=Depends(operator)):
    rows=[]
    for text,role,tools,kind,expect in CASES:
        r=check(text,policy,role,'selftest',tools)
        rows.append({'text':text,'kind':kind,'decision':r.decision,'risk':r.risk,'ok':r.decision in expect})
    return {'rows':rows,'passed':sum(r['ok'] for r in rows),'total':len(rows),'scope':'Тест правил без модели; не измеряет качество Qwen и не добавляет записи в журнал.'}

@app.get('/',response_class=HTMLResponse)
@app.get('/ui',response_class=HTMLResponse)
def ui():return (ROOT/'static/index.html').read_text(encoding='utf-8')

@app.post('/v1/check')
def contest_check(req: ContestRequest, who=Depends(identity)):
    start = time.perf_counter()
    if not who['admin']:
        req.context.user_role = who['role']
    try:
        response = inspect_request(req, policy.data['thresholds'])
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError):
        raise HTTPException(503, 'Проверка политики недоступна. Передача остановлена.')
    response['latency_ms'] = round((time.perf_counter() - start) * 1000, 3)
    logging.getLogger('uvicorn.error').info('contest_check decision=%s category=%s latency_ms=%s',
        response['decision'], response['risk_category'], response['latency_ms'])
    return response
