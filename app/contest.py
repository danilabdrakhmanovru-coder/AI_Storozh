"""Contest gateway. All inspected messages are data, including attached documents."""
import os
import re
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

from .contest_policy import parse_policy, read_policy
from .speech_act import speech, INFO
from pydantic import BaseModel, Field

from .classifier import classify
from .attachment_intent import detect as detect_attachment_intent
from .document_encoder import detect as detect_document_encoder
from .attachment_passage import detect as detect_attachment_passage
from .category_model import predict, risk_level
from .guard import COMPILED, normalize
from .jailbreak_intent import signals as jailbreak_signals
from .request_intent import assessment, PERSONA


ROLE_RULES = [(rid, regex) for rid, _, _, regex in COMPILED if rid in {'inj_roleplay', 'jb_dan'}]


class InferenceUnavailable(RuntimeError):
    """A required model failed; the caller must not forward the packet."""


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


def literal_list_edit(text, spec, specs):
    if spec.get('inline_literal') and re.fullmatch(spec['inline_literal'],text):
        return True
    match = spec.get('inline_repeated') and re.fullmatch(spec['inline_repeated'],text)
    if not match:
        return False
    words = re.findall(r'[a-zа-яё]+', match[1])
    # Treat only a repeated literal value as the operand. Any second action
    # in the value keeps the normal policy and permission checks.
    return len(set(words)) < len(words) and not any(spec["_match"].search(match[1]) for spec in specs.values())


CATEGORIES = {
    'prompt_extraction': 'system_prompt_extraction', 'instruction_tampering': 'system_prompt_override',
    'evasion': 'prompt_injection', 'secret_leak': 'unauthorized_data_access',
    'unauthorized_data': 'unauthorized_data_access', 'dangerous_command': 'tool_abuse',
}
EDUCATIONAL = re.compile(r'^(?:объясни|расскажи|что (?:такое|делает)|как работает|explain|what (?:is|does)|how does)\b', re.I)
REQUEST_ACTION = re.compile(
    r'\b(?:покажи|перешли|пришли|выгрузи|дай|открой|предоставь|передай|'
    r'составь|подготовь|придумай|создай|напиши|сгенерируй|сделай|show|send|export|write|create|draft|compose|prepare|design)\b')
FRAUDULENT_DOCUMENT = re.compile(
    r'^\s+(?:(?:a|an|the)\s+)?(?:fraudulent|counterfeit|forged|fake|bogus|deceptive)\s+'
    r'(?:(?:billing|payment|renewal|ticket|fare|delivery)\s+)?(?:invoice|notice|request|reminder|demand|letter|message)\b')
PAYMENT_INTENT = re.compile(
    r'\b(?:to (?:trick|deceive|divert)|(?:pays?|paying|send|wire|transfer|divert)\b.{0,45}'
    r'\b(?:impost[oe]r|scammer|fraudster|me|my account))\b')
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
        payment_fraud = FRAUDULENT_DOCUMENT.match(clause) and PAYMENT_INTENT.search(clause)
        credential_fraud = CREDENTIAL.search(clause) and DECEPTION.search(clause) and SOLICIT.search(clause)
        if payment_fraud or credential_fraud or FRAUD.search(clause):
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


def inspect_request(req, thresholds, raw_policy, *, encoder=None):
    # An explicit continuation inherits the previous user action, including
    # its role checks. Keep older turns and current attachments for inspection.
    if sum(len(x.content) for x in (*req.context.history,*req.context.attachments))+len(req.user_prompt)>96000:
        return result('block','tool_abuse','Суммарный контекст превышает 96 000 символов.')
    continued_index = None
    had_history = bool(req.context.history)
    history = req.context.history
    if history and history[-1].role == 'user' and CONTINUATION.fullmatch(req.user_prompt.strip()):
        continued_index = len(history)-1
        req = req.model_copy(update={'user_prompt': history[-1].content,
            'context': req.context.model_copy(update={'history': history[:-1]})})
    # A standalone user greeting carries no authorization or task. Keep all
    # other history, including attachments and quoted instructions, intact.
    history = req.context.history
    if len(history) == 1 and history[0].role == 'user' and GREETING.fullmatch(history[0].content.strip()):
        req = req.model_copy(update={'context': req.context.model_copy(update={'history': []})})
    if encoder is None and os.getenv('STOROZH_ENCODER_DIR'):
        from .inference import configured
        try:
            encoder = configured(os.environ['STOROZH_ENCODER_DIR'])
        except Exception as exc:
            raise InferenceUnavailable('Не удалось загрузить локальный ONNX-энкодер.') from exc
    policy = parse_policy(raw_policy)
    agent = policy['agents'].get(req.context.target_agent)
    if agent is None or req.context.user_role not in agent['roles']:
        return result('block','privilege_escalation','Агент или роль отсутствуют в действующей политике.')
    act = speech(req.user_prompt)
    mode = os.getenv('STOROZH_ATTACHMENT_INTENT')
    document_target = None
    review_vectors = {}
    if (encoder is None and os.getenv('STOROZH_VARIANT','B') == 'B' and mode == '13'
            and len(req.context.attachments) == 1 and not had_history
            and 'indirect_injection' in policy['always_block']):
        from .recipient_model import MAX_CHARS, read_only_task
        body=req.context.attachments[0].content
        if body.strip() and len(body) <= MAX_CHARS and read_only_task(req.user_prompt):
            from .recipient_calibration import classify as predict_recipient
            # Share one evaluation between the lexical rule and final review in this request.
            @lru_cache(maxsize=1)
            def document_target(text):
                return predict_recipient(text,review_vectors=review_vectors)
    def evaluate_packet(packet):
        verdict = _inspect(packet, thresholds, raw_policy, encoder, act, document_target)
        if verdict.pop('_recheck_prompt',False):
            clean=packet.model_copy(update={'context':packet.context.model_copy(update={'attachments':[]})})
            return evaluate_packet(clean)
        eligible = verdict.pop('_text_review',False)
        if eligible:
            from .confidence_review import reviewer
            revised=reviewer().revise(packet,verdict,policy['always_block'])
            if revised is not None:
                verdict=revised
        return verdict, eligible
    verdict, eligible = evaluate_packet(req)
    doc_model = verdict.pop('_document_model',False)
    if (encoder is None and os.getenv('STOROZH_VARIANT','B')=='B' and mode in ('11','12','13')
            and (req.context.attachments or eligible)):
        from .document_review import review
        def recheck_prompt():
            context = req.context.model_copy(update={'attachments':[]})
            return evaluate_packet(req.model_copy(update={'context':context}))[0]
        if mode in ('12','13') and req.context.attachments:
            from .recipient_model import review_attachment
            if mode=='13':
                from .recipient_calibration import classify
                verdict = review_attachment(req,verdict,policy,doc_model,recheck_prompt,
                                            document_target or classify,rule='document_recipient_v27',review_vectors=review_vectors)
            else:
                verdict = review_attachment(req,verdict,policy,doc_model,recheck_prompt)
        else:
            verdict = review(req,verdict,policy,recheck_prompt)
    verdict.pop('_document_model',None)
    if (encoder is None and mode=='13' and os.getenv('STOROZH_VARIANT','B')=='B'
            and agent.get('review_correspondence',False)):
        from .correspondence import review as review_correspondence
        verdict=review_correspondence(req,verdict,policy)
        if any(e.get('rule')=='correspondence_intent' for e in verdict.get('evidence',())):
            act='execute_request'
    verdict['speech_act'] = act
    if 'evidence' not in verdict:
        verdict['evidence'] = []
    if continued_index is not None:
        for item in verdict['evidence']:
            if item['source']=='prompt':
                item.update(source='history',index=continued_index)
    verdict['authority'] = {'agent':req.context.target_agent,'role':req.context.user_role}

    policy_level = verdict.pop('_operation_level', None)
    if encoder is None:
        verdict['risk_level'] = policy_level or risk_level(req, verdict['risk_category'])
    elif policy_level:
        verdict['risk_level'] = policy_level
    elif verdict['decision'] == 'block':
        # A hard rule owns the decision and category; the encoder estimates severity.
        try:
            level = encoder.evidence(req.user_prompt,req.context.model_dump())['risk_level']
        except Exception as exc:
            raise InferenceUnavailable('Ошибка оценки тяжести ONNX.') from exc
        if level != 'none':
            verdict['risk_level'] = level
    return verdict


GREETING = re.compile(r'(?:hello|hi|привет|здравствуйте)[.!]?', re.I)
CONTINUATION = re.compile(
    r'(?:продолж(?:и|ай)(?: по моей предыдущей просьбе)?|'
    r'выполни (?:мою )?(?:просьбу|запрос) из предыдущего сообщения|'
    r'(?:please )?(?:continue|go ahead|carry on)(?: with (?:my|the) (?:previous|last) request)?)'
    r'[.!]?', re.I)


def _inspect(req, thresholds, raw_policy, encoder=None, act=None, document_target=None):
    policy = parse_policy(raw_policy)
    ctx = req.context
    agent = policy['agents'].get(ctx.target_agent, {})
    role = agent.get('roles', {}).get(ctx.user_role, {})
    text = normalize(req.user_prompt)[0]
    specs = agent['tool_specs']
    educational = act in INFO
    informational_form = bool(re.search(r'(?:^|[?.!]\s+)(?:кто|зачем|почему|где|кому|чем|как|что (?:такое|значит)|объясни|расскажи|what|why|which|who|how|explain)\b', text))
    bypass_cue = bool(re.search(r'(?:отключи|сними|выключи|отмени|обойди|disable|drop|suspend|disregard)\b.{0,48}(?:огранич|защит|правил|запрет|safeguard|restriction|guardrail|safety)',text))
    if bypass_cue and 'jailbreak' in policy['always_block'] and (act=='bypass_attempt' or not educational):
        return result('block','jailbreak','Запрошено отключение ограничений защитника.')

    policy_definition = any(re.fullmatch(p,text,re.I) for p in agent['definition_patterns'])
    for source,index,content in [('attachment',i,a.content) for i,a in enumerate(ctx.attachments)] + [
        ('history',i,m.content) for i,m in enumerate(ctx.history)
    ] + [('prompt',0,req.user_prompt)]:
        views = normalize(content)
        if source != 'prompt':
            for forbid in agent['forbidden_actions']:
                if any(re.search(p,v,re.I) for p in forbid['patterns'] for v in views):
                    return dict(**result('block','indirect_injection','Недоверенный источник содержит запрещённое политикой действие.'), evidence=[dict(source=source,index=index,rule=forbid['name'])])
            if any(v['_match'].search(views[0]) for v in specs.values()) and speech(content)=='execute_request':
                return dict(**result('block','indirect_injection',f'Источник {source}: запрошен вызов инструмента из недоверенного контекста.'),evidence=[dict(source=source,index=index,rule='untrusted_tool')])
        if ('jailbreak'  in policy['always_block'] and re.search(
                r'\b(?:you are now|act as|pretend to be)\b.{0,85}\b(?:unfiltered|amoral|unrestricted)\b'
                r'.{0,60}\b(?:assistant|chatbot|agent|model)\b', views[0])):
            category = 'indirect_injection' if source == 'attachment' else 'jailbreak'
            return result('block', category, 'Запрошена роль ассистента без ограничений.')
        violation = requested_violation(views[0])
        if violation and (violation in policy['always_block'] or violation in policy['context_dependent']):
            category = 'indirect_injection' if source == 'attachment' else violation
            reason = ('запрошено создание обманного сообщения' if violation == 'fraud_social_engineering'
                      else 'запрошены чужие защищённые данные')
            return result('block', category, f'Источник {source}: {reason}.')
        for rid, category, weight, regex in COMPILED:
            if not any(regex.search(view) for view in views):
                continue
            if source == 'attachment' and rid == 'jb_dan' and document_target is not None:
                from .jailbreak_context import reference_candidate
                if reference_candidate(content,regex) and document_target(content) == 'reference':
                    continue
            if source=='prompt' and category=='dangerous_command' and policy_definition:
                continue
            category = CATEGORIES.get(category, category)
            if source == 'attachment' and category in policy['always_block']:
                category = 'indirect_injection'
            if source == 'prompt' and educational and informational_form and rid in {'jb_dan', 'jb_nolimits','rce_pipe'} and not bypass_cue and not re.search(r'[|;&<>]',text):
                continue
            if category in policy['always_block'] or (category == 'unauthorized_data_access'
                    or (category == 'tool_abuse' and weight >= 0.7)):
                # A generic override can introduce a request for a persona
                # without safeguards. Resolve its type; both decisions block.
                if (encoder is None and source == 'prompt' and not ctx.attachments and not ctx.history
                        and not educational and 'jailbreak' in policy['always_block']
                        and (category in {'prompt_injection', 'system_prompt_override'}
                             or (rid == 'priv_claim' and PERSONA.search(text)))):
                    try:
                        intent = assessment(req)
                    except Exception as exc:
                        raise InferenceUnavailable('Модель намерения недоступна.') from exc
                    minimum = 1.0 if rid == 'priv_claim' else 0.60
                    if intent['jailbreak'] >= minimum and max(intent['ordinary'], intent['other']) < -0.30:
                        return result('block', 'jailbreak', f'Правило {rid} и модель выявили попытку снять защиту.')
                if (source == 'prompt' and not ctx.attachments and not educational
                        and category in {'prompt_injection', 'system_prompt_override'}
                        and 'jailbreak' in policy['always_block']
                        and {'role', 'reply'} <= jailbreak_signals(text)):
                    for role_rule, pattern in ROLE_RULES:
                        if pattern.search(text):
                            return result('block', 'jailbreak',
                                          f'Запрошена смена роли с обходом защиты. Правила {role_rule} и {rid}.')
                hit=regex.search(content)
                detail=dict(source=source,index=index,rule=rid)
                if hit:detail.update(start=hit.start(),end=hit.end())
                return dict(**result('block',category,f'Сработало правило {rid}, источник {source}.'),evidence=[detail])
    # All sources have passed explicit denials and the untrusted tool checks.
    # A configured action can depend on the presence of an attachment, so
    # always leave those packets on the complete authorization path.
    if document_target is not None and ctx.attachments:
        policy_operand=(any(spec['_match'].search(text) for spec in specs.values())
                        or any(re.search(p,text,re.I) for action in agent['forbidden_actions'] for p in action['patterns'])
                        or any(re.search(item['pattern'],text,re.I) for item in agent['protected_data']))
        if not policy_operand:
            target=document_target(ctx.attachments[0].content)
            if target=='agent':
                return dict(**result('block','indirect_injection',
                    'Вложенный документ адресует команду текущему ассистенту.'),
                    evidence=[dict(source='attachment',index=0,rule='document_recipient_v27')])
            if target=='reference':
                return {'_recheck_prompt':True}
    semantic_req = req
    if (ctx.attachments and encoder is None and os.getenv('STOROZH_VARIANT','B')=='B'
            and os.getenv('STOROZH_ATTACHMENT_INTENT') in ('1','2','3','4','5','6','7','8','9','10','11','12','13')
            and 'indirect_injection' in policy['always_block']):
        mode = os.environ['STOROZH_ATTACHMENT_INTENT']
        if mode in ('12','13'): mode='11'
        if mode in ('6','7','8','9','10','11') or (mode in ('4','5') and any(len(a.content)>900 for a in ctx.attachments)):
            from .attachment_cascade import evaluate as check_documents
            hit, evidence_docs = check_documents(ctx.attachments, learned=mode in ('5','6','7','8','9','10','11'), isolated=mode in ('6','7','8','9','10','11'))
            if not hit:
                semantic_req = req.model_copy(update={'context':ctx.model_copy(update={'attachments':evidence_docs})})
        elif mode == '1':
            hit = detect_attachment_intent(ctx.attachments)
        else:
            hit = detect_attachment_passage(ctx.attachments) if mode in ('3','4','5') else None
            if not hit:
                hit = detect_document_encoder(ctx.attachments)
        if hit:
            return dict(**result('block','indirect_injection',
                        f"Вложение №{hit['index']+1} просит изменить ответ агента. Политика блокирует такую инструкцию."),
                        evidence=[hit],_document_model=True)
    for action in agent['forbidden_actions']:
        if any(re.search(p,text,re.I) for p in action['patterns']) and not (policy_definition):
            return result('block','tool_abuse',f"Запрет {action['name']}: {action['description']}.")
    for protected in agent['protected_data']:
        if not re.search(protected['pattern'],text,re.I):
            continue
        mode=role.get(protected['permission'],'deny')
        if mode=='allow':continue
        if mode!='names_only' or not re.search(protected['metadata_pattern'],text,re.I):
            return result('block','unauthorized_data_access','Политика роли запрещает доступ к запрошенным данным.')
        if not ctx.history and not ctx.attachments and re.fullmatch(protected['metadata_literal'],text.strip(),re.I):
            return result('allow','none','Политика роли разрешает чтение только названий.')
    decisions = []
    command_text = re.sub(r'^(?:please|пожалуйста),?\s+', '', text.strip())
    literal_tool = next((name for name,spec in specs.items() if any(p.fullmatch(command_text) for p in spec['_literal'])),None)
    words = re.findall(r'[\w-]+',text)
    envs = agent['environments']
    matched = [e for e,globs in envs.items() if any(w == e or w in agent['environment_aliases'].get(e,[])
               or any(fnmatchcase(w,g) for g in globs) for w in words)]
    # A literal policy command and an explicit alias imperative provide execution evidence
    # even on previously unseen tool vocabularies.
    for tool,spec in specs.items():
        match = spec['_match'].search(text)
        prefix = text[:match.start()].strip() if match else ''
        direct_alias = match and (not prefix or prefix.endswith(('.','!',';')) or (matched and prefix.endswith(',') and not informational_form)) and not text.endswith('?')
        if direct_alias and not prefix and matched and re.fullmatch(r'[\s:]+[\w-]+[.!]?',text[match.end():]):
            literal_tool = tool
        # A recognized action in an otherwise ambiguous utterance still needs an ACL decision.
        # Suppress mentions only when the utterance also has affirmative discussion evidence.
        discussion = educational and (informational_form or re.search(
            r'\b(?:вчера|сообщил[аи]?|зафиксирован[ао]?|сводк[ае]|отч[её]те|yesterday|recorded|reported|documentation|учебни\w*)\b',text))
        if (discussion and not direct_alias and not literal_tool) or not (tool==literal_tool or match):
            continue
        if spec['type']=='destroy' and not ctx.history and not ctx.attachments and literal_list_edit(text.strip(),spec,specs):
            continue
        rule = role['tools'].get(tool,'deny')
        for env in matched or list(envs):
            choice = access(rule,env)
            decisions.append((choice,tool,env,f"Роль {ctx.user_role}, инструмент {tool}, окружение {env}. Правило доступа: {choice}."))
    literal_operation = bool(decisions and literal_tool and not ctx.history and not ctx.attachments)
    if decisions:
        decision,tool,env,reason = max(decisions,key=lambda row: (
            ['allow','confirm','block'].index(row[0]),specs[row[1]]['type']=='destroy',row[2] in agent['critical_environments']))
        access_result = result(decision,{'allow':'none','confirm':'sensitive_action','block':'tool_abuse'}[decision],reason)
        critical = env in agent['critical_environments'] and specs[tool]['type'] in ('modify','destroy')
        if decision=='block' or literal_operation or encoder is not None or (len(set(matched))==1 and specs[tool]['type']=='destroy'):
            access_result['_operation_level'] = {'block':'critical' if critical else 'high','confirm':'high' if critical and specs[tool]['type']=='destroy' else 'medium','allow':'none'}[decision]
    discussion_vote = False
    cleared_categories = set()
    semantic_category = None
    current_head = os.getenv('STOROZH_ATTACHMENT_INTENT')
    if current_head in ('12','13'): current_head='11'
    def text_vote(verdict):
        if (current_head in ('10','11') and os.getenv('STOROZH_VARIANT','B')=='B' and encoder is None
                and not decisions and not ctx.history and not ctx.attachments and not bypass_cue):
            verdict['_text_review']=True
        return verdict
    if (current_head in ('8','9') and encoder is None and os.getenv('STOROZH_VARIANT','B')=='B'
            and not decisions and not ctx.history and not ctx.attachments):
        from .semantic_review import reviewer, INJECTION
        clear_content = False
        if current_head=='9':
            from .intent_review import reviewer as intent_reviewer
            ordinary, candidate, clear_content = intent_reviewer().review(req.user_prompt)
        else:
            ordinary, candidate = reviewer().assess(req.user_prompt)
        if ordinary and not bypass_cue:
            cleared_categories = INJECTION
            if clear_content:
                cleared_categories = cleared_categories | {'harmful_content','abuse','fraud_social_engineering'}
        if candidate in policy['always_block']:
            semantic_category = candidate
    if (encoder is None and os.getenv('STOROZH_VARIANT','B')=='B'
            and current_head in ('7','8','9','10','11') and not (decisions and decision=='block')):
        from .injection_intent import detect
        signal, discussion_vote = detect(semantic_req, policy['always_block'], req)
        if signal and signal[0] not in cleared_categories:
            category, detail = signal
            return text_vote(dict(**result('block',category,f"Модель обнаружила попытку изменить задачу или границы доступа. Категория {category}, источник {detail['source']}."),
                        evidence=[detail],_document_model=detail['source']=='attachment'))
    if encoder is not None:
        if decisions and (decision == 'block' or literal_operation):
            return access_result
        try:
            semantic = encoder.evidence(req.user_prompt,ctx.model_dump())
            lo, hi = encoder.config['confirm'], encoder.config['block']
        except Exception as exc:
            raise InferenceUnavailable('Ошибка проверки ONNX.') from exc
        cat, prob = semantic['category'], semantic['danger_prob']
        if cat in ('none','sensitive_action') and prob >= hi:
            cat = semantic['alternative']
        if cat in policy['always_block']:
            choice = 'block'
        elif decisions:
            return access_result
        elif cat in policy['context_dependent']:
            # Cannot establish the object or the required permission.
            choice = 'confirm'
        elif prob >= lo or cat == 'sensitive_action':
            choice = 'confirm'
        else:
            choice = 'allow'
        out = result(choice, cat, f'Энкодер: {cat}, оценка опасности {prob:.3f}; решение {choice}.')
        if choice == 'allow':
            out['risk_level'] = 'none'
        elif semantic['risk_level'] != 'none':
            out['risk_level'] = semantic['risk_level']
        out['_operation_level'] = out['risk_level']
        return out
    try:
        prioritize_jailbreak = 'jailbreak' in policy['always_block']
        category = predict(semantic_req, check_danger=not decisions, prioritize_jailbreak=prioritize_jailbreak)
        if ctx.attachments and category in policy['always_block'] and category != 'indirect_injection':
            for attachment in semantic_req.context.attachments:
                doc = req.model_copy(update={'user_prompt':attachment.content,
                    'context':ctx.model_copy(update={'history':[], 'attachments':[]})})
                if predict(doc, prioritize_jailbreak=prioritize_jailbreak) == category:
                    return dict(**result('block', 'indirect_injection', f'Во вложении обнаружена инструкция категории {category}.'),_document_model=True)
    except Exception as exc:
        raise InferenceUnavailable('Модель категорий недоступна.') from exc
    score = classify(req.user_prompt)
    if score['status'] != 'ok':
        raise InferenceUnavailable('Классификатор недоступен.')
    if literal_operation:
        return access_result
    if category in cleared_categories:
        category = None
    if semantic_category and category not in policy['always_block'] and category not in policy['context_dependent']:
        return dict(**result('block',semantic_category,f'Смысловой классификатор выявил запрещённое намерение: {semantic_category}.'),
                    evidence=[dict(source='prompt',index=0,rule='intent_review_v22' if current_head=='9' else 'semantic_review_v21')])
    if (discussion_vote and not decisions and not ctx.history and not ctx.attachments
            and not bypass_cue and category in {'jailbreak','prompt_injection','system_prompt_extraction','system_prompt_override'}
            and score['risk'] < thresholds['block']):
        return text_vote(result('allow','none','Модель распознала вопрос о механизме или порядке работы.'))
    if (educational and informational_form and not bypass_cue and not ctx.history and not ctx.attachments
            and not decisions and category in {None,'none','tool_abuse','sensitive_action'}
            and score['risk'] < thresholds['block']):
        return text_vote(result('allow','none','Вопрос о понятии или порядке работы. Выполнение действия не запрошено.'))
    if category in policy['always_block'] or (not decisions and category in policy['context_dependent']):
        verdict=result('block', category, f'Модель v14 определила категорию {category}. Политика запрещает такие запросы.')
        if ctx.attachments and not decisions:
            verdict['_document_model']=True
        return text_vote(verdict)
    if decisions:
        if decision == 'block' or score['risk'] < thresholds['block']:
            return access_result
    if score['risk'] >= thresholds['block'] and 'prompt_injection' not in cleared_categories:
        return text_vote(result('block', 'prompt_injection', f"Классификатор выявил риск ({score['risk']:.2f})."))
    return text_vote(result('allow', 'none', 'Проверки текста и контекста не выявили нарушения политики.'))
