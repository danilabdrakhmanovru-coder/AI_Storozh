"""ИИ-сторож: API and local operator console. Run with uvicorn app.main:app."""
import yaml
from .contest import ContestRequest, InferenceUnavailable, inspect_request, read_policy, parse_policy
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
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from .datasets import parse_dataset, predictions, MAX_BYTES
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exception_handlers import http_exception_handler
from . import contest_policy as policy_files
from .contest_worker import BUDGET_SECONDS, BudgetExceeded, worker as cpu_worker
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

def bounded_contest_mode():
    return os.getenv('STOROZH_VARIANT','B')=='B' and os.getenv('STOROZH_ATTACHMENT_INTENT') in ('4','5','6','7','8','9','10','11','12','13')

@app.on_event('startup')
def warm_cpu_worker():
    if bounded_contest_mode():cpu_worker.warm()

@app.on_event('shutdown')
def close_cpu_worker():
    cpu_worker.close()

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
def contest_ui():
    return (ROOT/'static/agent-test.html').read_text(encoding='utf-8')

@app.get('/ui',response_class=HTMLResponse)
def ui():return (ROOT/'static/index.html').read_text(encoding='utf-8')

def contest_policy():
    raw = read_policy()
    return raw, parse_policy(raw)['thresholds'], hashlib.sha256(raw).hexdigest()


def unavailable(message, start=None, version=None):
    return dict(decision='block',risk_level='high',risk_category='tool_abuse',explanation=message,
                state='blocked',policy_version=version,check_error=True,
                latency_ms=round((time.perf_counter()-start)*1000,3) if start else 0.0)


@app.exception_handler(RequestValidationError)
async def validation_failure(request, exc):
    if request.url.path == '/v1/check':
        return JSONResponse(unavailable('Некорректный запрос или контекст. Проверка остановлена.'))
    return await request_validation_exception_handler(request,exc)


@app.exception_handler(StarletteHTTPException)
async def invalid_body(request,exc):
    if request.url.path == '/v1/check' and exc.status_code==400:
        return JSONResponse(unavailable('Не удалось прочитать JSON в кодировке UTF-8.'))
    return await http_exception_handler(request,exc)


class PolicyEdit(BaseModel):
    yaml_text: str = Field(max_length=262144)
    version: str = Field(min_length=64,max_length=64)


def public_schema():
    raw, _, version = contest_policy()
    data = parse_policy(raw)
    agents={k:{'roles':list(v['roles']),'environments':v['environments'],'tools':v['tools']} for k,v in data['agents'].items()}
    return dict(version=version,agents=agents)


@app.get('/v1/schema')
def contest_schema(who=Depends(identity)):
    return public_schema()


@app.get('/v1/policy')
def get_contest_policy(who=Depends(operator)):
    raw, _, version = contest_policy()
    with store.lock:
        last=store.db.execute('SELECT author,ts,version FROM contest_policy_events ORDER BY rowid DESC LIMIT 1').fetchone()
    return dict(yaml_text=raw.decode('utf-8-sig'),version=version,agents={k:v['roles'] for k,v in parse_policy(raw)['agents'].items()},
                changed=dict(last) if last else None)


def writable_policy(who=Depends(operator)):
    if who['demo']:
        raise HTTPException(403,'Для изменения политики задайте ключ оператора STOROZH_API_KEY.')
    return who


@app.post('/policy/reload')
def reload_contest_policy(change: Optional[PolicyEdit]=None, who=Depends(writable_policy)):
    with policy_files.LOCK:
        try:
            if change:policy_files.apply_policy(change.yaml_text.encode(),change.version)
            raw=policy_files.read_policy(strict=True)
        except Exception:
            raise HTTPException(422,'Политика отклонена. Проверьте YAML, права, шаблоны и версию документа.')
        version=hashlib.sha256(raw).hexdigest()
        with store.lock, store.db:
            store.db.execute('INSERT INTO contest_policy_events VALUES (?,?,?)',(who['user_id'],time.time(),version))
        return {'version':version}


def packet_fingerprint(req, who):
    if not who['admin']:
        req.context.user_role = who['role']
        req.context.user_id = who['user_id']
    packet = json.dumps(req.model_dump(), sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(packet).hexdigest()


@app.post('/v1/check')
def contest_check(req: ContestRequest, who=Depends(identity)):
    start = time.perf_counter()
    deadline = time.monotonic() + BUDGET_SECONDS
    pv = None
    try:
        raw, thresholds, pv = contest_policy()
        if not who['admin'] and req.context.user_role != who['role']:
            return unavailable('Роль в контексте расходится с ролью удостоверения.',start,pv)
        fingerprint = packet_fingerprint(req, who)
        if bounded_contest_mode():
            response = cpu_worker.check(req.model_dump(), thresholds, raw, deadline)
        else:
            response = inspect_request(req, thresholds, raw)
    except BudgetExceeded:
        return unavailable('Истёк бюджет проверки. Передача запрещена.',start,pv)
    except Exception:
        return unavailable('Защитник не завершил проверку. Запрос заблокирован.',start,pv)
    rid = uuid.uuid4().hex
    state = {'allow': 'ready', 'confirm': 'pending', 'block': 'blocked'}[response['decision']]
    latency = round((time.perf_counter() - start) * 1000, 3)
    role = req.context.user_role
    try:
        with store.lock, store.db:
            store.db.execute('INSERT INTO contest_checks VALUES (?,?,?,?,?,?,?,?,?,?,NULL)',
                             (rid, time.time(), who['user_id'], role if role in parse_roles(raw) else 'unknown',
                              response['decision'], response['risk_category'], latency, pv, fingerprint, state))
    except Exception:
        return unavailable('Не удалось записать решение в журнал.',start,pv)
    return {**response, 'latency_ms': latency, 'request_id': rid, 'state': state, 'policy_version': pv}


def parse_roles(raw):
    return {role for agent in parse_policy(raw)['agents'].values() for role in agent['roles']}


def transition(rid, req, who, confirm):
    fingerprint = packet_fingerprint(req, who)
    _, _, pv = contest_policy()
    expected = ('pending',) if confirm else ('ready', 'approved')
    state = 'approved' if confirm else 'delivered'
    now = time.time()
    # The local receiver records one receipt in the same transaction as delivery.
    with store.lock, store.db:
        changed = store.db.execute("""
            UPDATE contest_checks SET state=?, received_at=?
            WHERE id=? AND owner=? AND fingerprint=? AND policy_version=? AND ts>?
              AND state IN (?,?)
            """, (state, None if confirm else now, rid, who['user_id'], fingerprint, pv,
                  now - 3600, expected[0], expected[-1]))
        if changed.rowcount != 1:
            raise HTTPException(409, 'Действие недоступно. Проверьте запрос и контекст заново.')
    return {'request_id': rid, 'state': state, 'received_at': None if confirm else now}


@app.post('/v1/demo/{rid}/confirm')
def contest_confirm(rid: str, req: ContestRequest, who=Depends(identity)):
    return transition(rid, req, who, confirm=True)


@app.post('/v1/demo/{rid}/deliver')
def contest_deliver(rid: str, req: ContestRequest, who=Depends(identity)):
    return transition(rid, req, who, confirm=False)


@app.get('/v1/demo/journal')
def contest_journal(decision: Optional[str]=None,category: Optional[str]=None,role: Optional[str]=None,
                    since: Optional[float]=None,until: Optional[float]=None,who=Depends(identity)):
    conditions=['owner=?']
    values=[who['user_id']]
    for clause,value in [('decision=?',decision),('category=?',category),('role=?',role),('ts>=?',since),('ts<=?',until)]:
        if value is not None:
            conditions.append(clause)
            values.append(value)
    with store.lock:
        rows=store.db.execute("SELECT id,ts,role,decision,category,latency_ms,policy_version,state,received_at FROM contest_checks WHERE "
            +" AND ".join(conditions)+" ORDER BY ts DESC LIMIT 100",values).fetchall()
    return [dict(row) for row in rows]


def dataset_access(request: Request, x_api_key: Optional[str] = Header(None)):
    key = os.getenv('STOROZH_DATASET_KEY')
    if key and secrets.compare_digest(x_api_key or '', key):
        origin = request.headers.get('origin')
        if origin and urlparse(origin).netloc != request.headers.get('host'):
            raise HTTPException(403, 'Запрос с другого сайта запрещён')
        return
    operator(identity(request, x_api_key))


@app.post('/v1/datasets/check', response_class=StreamingResponse, openapi_extra={
    'requestBody': {'required': True, 'content': {'application/octet-stream': {
        'schema': {'type':'string','format':'binary'}}}},
    'responses': {'200': {'description':'Последовательность предсказаний JSONL',
                           'content': {'application/x-ndjson': {}}}}})
async def dataset_check(request: Request, format: Literal['jsonl','csv','zip']=Query('jsonl'), report: bool=False, access=Depends(dataset_access)):
    """Проверка загруженного набора. Роли в файле задаёт доверенный оператор."""
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, 'Файл превышает 32 МиБ.')
        data.extend(chunk)
    packets = await run_in_threadpool(parse_dataset, data, format, report)
    del data
    raw, thresholds, pv = contest_policy()
    return StreamingResponse(predictions(packets, raw, thresholds), media_type='application/x-ndjson',
        headers={'X-Total-Rows':str(len(packets)), 'X-Policy-Version':pv,
                 'Cache-Control':'no-store', 'X-Accel-Buffering':'no',
                 'Content-Disposition':'attachment; filename="predictions.jsonl"'})
