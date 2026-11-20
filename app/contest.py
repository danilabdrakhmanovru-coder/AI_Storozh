"""Contest gateway. All inspected messages are data, including attached documents."""
import os
import re
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .classifier import classify
from .category_model import predict
from .guard import COMPILED, normalize


class PacketPart(BaseModel):
    model_config = {"extra": "allow"}


class Message(PacketPart):
    role: str
    content: str = Field(max_length=32000)


class Attachment(PacketPart):
    content: str = Field(max_length=32000)


class Context(PacketPart):
    target_agent: str
    user_role: str
    history: list[Message] = Field(default_factory=list, max_length=64)
    attachments: list[Attachment] = Field(default_factory=list, max_length=16)


class ContestRequest(BaseModel):
    user_prompt: str = Field(min_length=1, max_length=32000)
    context: Context


@lru_cache(maxsize=2)
def parse_policy(raw):
    return yaml.safe_load(raw)


TOOLS = {
    'restart_service': r'\b(?:перезапусти(?:те)?|рестартни|restart|restart_service|(?:инициируй|выполни|сделай) перезапуск)\b',
    'delete_resource': r'\b(?:удали(?:те)?|delete|remove|delete_resource|(?:запрашиваю|выполни|инициируй) удаление)\b',
    'read_logs': r'(?:покажи|прочитай|выведи|посмотри|read|show|fetch).{0,40}(?:лог|журнал|logs)|\bread_logs\b',
    'run_shell': r'(?:выполни|запусти|run|execute).{0,35}(?:команд|скрипт|shell|command|script|миграци|migration)|\b(?:run|выполни)\s+`?(?:df|du|ps|uptime|date)\b|\brun_shell\b|\b(?:масштабируй|deploy|scale|roll ?back|очисти кэш|поставь log_level)\b',
}
FORBIDDEN = {
    'disable_audit_or_wipe_logs': r'(?:отключи|сотри|очисти|disable|wipe|clear).{0,30}(?:аудит|журнал|логи|audit|logs)',
    'pipe_remote_script_to_shell': r'(?:curl|wget)\b[^|]{0,200}\|\s*(?:ba)?sh',
    'add_ssh_keys_or_sudoers_via_chat': r'(?:добавь|запиши|add|append).{0,80}(?:ssh|sudoers|authorized_keys)',
    'send_secrets_or_data_to_external_hosts': r'(?:send|upload|отправь|загрузи).{0,80}(?:https?://|внешн|external)',
}
CATEGORIES = {
    'prompt_extraction': 'system_prompt_extraction', 'instruction_tampering': 'system_prompt_override',
    'evasion': 'prompt_injection', 'secret_leak': 'unauthorized_data_access',
    'unauthorized_data': 'unauthorized_data_access', 'dangerous_command': 'tool_abuse',
}
DIRECTIVE = re.compile(r'\b(?:перезапусти|выполни|отправь|удали|покажи|запусти|игнорируй|execute|delete|send|ignore)\b', re.I)
EDUCATIONAL = re.compile(r'^(?:объясни|расскажи|что (?:такое|делает)|как работает|explain|what (?:is|does)|how does)\b', re.I)
REQUEST_ACTION = re.compile(
    r'\b(?:покажи|перешли|пришли|выгрузи|дай|открой|предоставь|передай|'
    r'составь|подготовь|придумай|создай|напиши|сгенерируй|сделай|show|send|export|write|create|draft|design)\b')
FRAUD = re.compile(
    r'\b(?:фишингов\w*\s+(?:письм|сообщени|страниц)|(?:выманить|вымани|обманом получить)'
    r'.{0,65}(?:парол|код|токен|деньги)|phishing\s+(?:email|message|page))')
CREDENTIAL = re.compile(r'парол|одноразов\w+ код|код\w* (?:mfa|из смс)|password|credentials|one.time code')
DECEPTION = re.compile(r'фейков|поддельн|обманом|притвор|выдавая себя|от имени|fake|impersonat|posing as')
SOLICIT = re.compile(r'собрать|собирать|выведать|ввести|ввели|продиктовать|прислать|'
                     r'collect|harvest|enter|share|disclose|provide')
PRIVATE = re.compile(r'\b(?:чуж\w+|без (?:\w+ ){0,2}(?:согласия|разрешения)|'
                     r'другого (?:пациента|клиента|пользователя)|without (?:their )?(?:permission|consent))\b')
RECORD = re.compile(r'выписк|сч[её]т|транзакци|платеж|медицинск|карт[ауые]|заключени|анализ'
                    r'|диагноз|истори|переписк|обращени|тикет|журнал|запис|данны|record|transcript')


def requested_violation(text):
    for action in REQUEST_ACTION.finditer(text):
        # Evaluate each requested action separately from a preceding explanation or refusal.
        clause = re.split(r'[.!?;\n]', text[action.end():action.end() + 240], maxsplit=1)[0]
        if re.match(r'\s*[, ]*(?:памятку|предупреждение|отказ|вымышленный пример|учебный разбор|'
                    r'как (?:распознать|обнаружить|предотвратить)|how to (?:detect|report|prevent))\b', clause):
            continue
        if FRAUD.search(clause) or (CREDENTIAL.search(clause) and DECEPTION.search(clause)
                                    and SOLICIT.search(clause)):
            return 'fraud_social_engineering'
        if PRIVATE.search(clause) and RECORD.search(clause):
            return 'unauthorized_data_access'
    return None


def result(decision, category, reason):
    level = {'allow': 'none', 'confirm': 'low', 'block': 'high'}[decision]
    return dict(decision=decision, risk_level=level, risk_category=category, explanation=reason)


def access(rule, env):
    if isinstance(rule, dict):
        rule = rule.get(env, rule.get('any', 'deny'))
    if rule in ('allow', 'any', env):
        return 'allow'
    return 'confirm' if rule == 'confirm' else 'block'


def read_policy():
    path = Path(os.getenv('ASTRA_POLICY_PATH', Path(__file__).resolve().parents[1] / 'security_policy.yaml'))
    return path.read_bytes()


def inspect_request(req, thresholds, raw_policy):
    policy = parse_policy(raw_policy)
    ctx = req.context
    agent = policy['agents'].get(ctx.target_agent, {})
    role = agent.get('roles', {}).get(ctx.user_role, {})
    text = normalize(req.user_prompt)[0]
    educational = EDUCATIONAL.search(text) and not DIRECTIVE.search(text)
    for source, content in [('attachment', a.content) for a in ctx.attachments] + [
        ('history', m.content) for m in ctx.history if m.role != 'system'
    ] + [('prompt', req.user_prompt)]:
        views = normalize(content)
        violation = requested_violation(views[0])
        if violation and (violation in policy['always_block'] or violation in policy['context_dependent']):
            category = 'indirect_injection' if source == 'attachment' else violation
            reason = ('запрошено создание обманного сообщения' if violation == 'fraud_social_engineering'
                      else 'запрошены чужие защищённые данные')
            return result('block', category, f'Источник {source}: {reason}.')
        for rid, category, weight, regex in COMPILED:
            if not any(regex.search(view) for view in views):
                continue
            if (rid == 'rce_pipe' and source == 'prompt' and re.fullmatch(
                    r'(?:hi[!,]?\s*)?explain why (?:curl|wget)\s*\|\s*(?:ba)?sh '
                    r'(?:installs? )?(?:is|are) (?:dangerous|unsafe|risky)[.?! ]*(?:please[.! ]*)?', text)):
                continue
            if (rid == 'rce_pipe' and source == 'prompt' and
                    re.fullmatch(r'(?:что (?:такое|означает)|what (?:is|does)) (?:термин |the term )?'
                                 r'reverse shell(?: (?:в (?:отч[её]те|документации|логах)(?: об инциденте)?|mean))?[?.! ]*', text)):
                continue
            category = CATEGORIES.get(category, category)
            if source == 'attachment' and category in policy['always_block']:
                category = 'indirect_injection'
            if source == 'prompt' and educational and rid in {'jb_dan', 'jb_nolimits'}:
                continue
            if category in policy['always_block'] or (category == 'unauthorized_data_access'
                    or (category == 'tool_abuse' and weight >= 0.7)):
                return result('block', category, f'Сработало правило {rid}, источник {source}.')
    for action in agent.get('forbidden_actions', []):
        if action not in FORBIDDEN:
            raise ValueError('Неизвестное запрещённое действие')
        if re.search(FORBIDDEN[action], text):
            return result('block', 'tool_abuse', f'Действие запрещено политикой: {action}.')
    if re.search(r'(покажи|выведи|прочитай|list|show|read).{0,45}(парол|секрет|токен|secret|password|token)', text):
        names = re.search(r'только (?:имена|названия)|names only|only (?:the )?names', text)
        if role.get('secrets') != 'names_only' or not names:
            return result('block', 'unauthorized_data_access', 'Политика роли запрещает запрошенный доступ к секретам.')
    decisions = []
    if not educational:
        words = re.findall(r'[\w-]+', text)
        environments = agent.get('environments', {})
        matched = [env for env, patterns in environments.items()
                   if env in words or any(fnmatchcase(w, p) for w in words for p in patterns)]
        if 'production' in words or re.search(r'\bпрод(?:е|акш\w*)?\b', text):
            matched.append('prod')
        if 'nonprod' in environments and {'dev', 'stage', 'test', 'staging', 'development'} & set(words):
            matched.append('nonprod')
        for tool, pattern in TOOLS.items():
            if re.search(pattern, text):
                rule = role.get('tools', {}).get(tool, 'deny') if tool in agent.get('tools', []) else 'deny'
                for env in matched or list(environments) or ['unknown']:
                    decision = access(rule, env)
                    decisions.append((decision, f'{ctx.user_role}, {tool}, {env}: {decision}.'))
    try:
        category = predict(req, check_danger=not decisions)
    except Exception:
        return result('block', 'tool_abuse', 'Модель категорий недоступна. Передача остановлена.')
    if category in policy['always_block'] or (not decisions and category in policy['context_dependent']):
        return result('block', category, f'Модель v4 определила категорию {category}. Политика запрещает такие запросы.')
    score = classify(req.user_prompt)
    if score['status'] != 'ok':
        return result('block', 'tool_abuse', 'Классификатор недоступен, передача остановлена.')
    if decisions:
        decision, reason = max(decisions, key=lambda item: ['allow', 'confirm', 'block'].index(item[0]))
        category = {'allow': 'none', 'confirm': 'sensitive_action', 'block': 'tool_abuse'}[decision]
        if decision == 'block' or score['risk'] < thresholds['block']:
            return result(decision, category, reason)
    if score['risk'] >= thresholds['block']:
        return result('block', 'prompt_injection', f"Классификатор выявил риск ({score['risk']:.2f}).")
    return result('allow', 'none', 'Проверки текста и контекста не выявили нарушения политики.')
