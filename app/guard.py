"""Ядро ИИ-сторожа: нормализация -> правила -> политика -> решение."""
import base64, codecs, threading, urllib.parse, re, unicodedata, json, time, os
from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
HOMOGLYPHS = str.maketrans("асеорхуіјԁѕ", "aceopxyijds")  # кириллица -> латиница
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
HOMOGLYPHS_RU = str.maketrans("aceopxyi", "асеорхуі")     # латиница -> кириллица (смешанные слова)


def normalize(text: str) -> list[str]:
    """Возвращает варианты текста: исходный, с гомоглифами, с раскодированным base64."""
    t = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH).lower()
    t = re.sub(r"\s+", " ", t)
    views = [t, t.translate(HOMOGLYPHS), t.translate(HOMOGLYPHS_RU), t.translate(LEET),
             re.sub(r"(?<=\b\w)[\s.\-_*|](?=\w\b)", "", t)]      # «i g n o r e», «и.г.н.о.р.и.р.у.й»
    if "%" in t:
        views.append(urllib.parse.unquote(t))
    views.append(codecs.decode(t, "rot13"))
    for chunk in re.findall(r"[A-Za-z0-9+/=]{16,}", text):
        try:
            dec = base64.b64decode(chunk + "=" * (-len(chunk) % 4), validate=False).decode("utf-8")
            if dec.isprintable():
                views.append(dec.lower())
        except Exception:
            pass
    return views


# (id, category, weight, regex)
RULES = [
    ("inj_ignore_en", "prompt_injection", 0.85, r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your)\s+(instructions?|rules?|prompts?|guidelines?)"),
    ("inj_ignore_ru", "prompt_injection", 0.85, r"(игнорируй|забудь|отбрось|не учитывай|отмени)\s+(все\s+|любые\s+)?(предыдущие|прежние|прошлые|свои|прошлых)?\s*(инструкци|правил|указани|ограничени)"),
    ("inj_roleplay", "jailbreak", 0.7, r"(you are now|ты теперь|act as|притворись|представь,? что ты)\s+.{0,40}(без ограничений|no restrictions|unrestricted|dan\b|без цензуры|evil|злой)"),
    ("jb_dan", "jailbreak", 0.8, r"\bdan\b.{0,30}(mode|режим)|developer mode|режим разработчика|do anything now|jailbreak|джейлбрейк"),
    ("jb_nolimits", "jailbreak", 0.6, r"без (каких-либо )?(ограничений|цензуры|фильтров)|without (any )?(restrictions|filters|limitations)"),
    ("sys_extract_en", "prompt_extraction", 0.8, r"(show|reveal|print|repeat|output|leak|display)\s+.{0,20}(system|initial|hidden|original)\s+(prompt|instructions?|message)"),
    ("sys_extract_ru", "prompt_extraction", 0.8, r"(покажи|выведи|раскрой|повтори|пришли|напиши|перечисли)\s+.{0,40}(системн\w+\s+(промпт|инструкци\w+|сообщени\w+)|свои\s+инструкци\w+|скрыт\w+\s+инструкци\w+|начальн\w+\s+(промпт|инструкци\w+))"),
    ("sys_fake_tag", "instruction_tampering", 0.8, r"<\|im_start\|>|\[/?inst\]|<<sys>>|<system>|\[system\]|^system\s*:|###\s*system"),
    ("sys_new_instr", "instruction_tampering", 0.55, r"(new|updated|новые|обновл\w+)\s+(system\s+)?(instructions?|инструкци\w+)\s*:"),
    ("secret_files", "sensitive_access", 0.8, r"/etc/(passwd|shadow)|\.ssh/|id_rsa|\.aws/credentials|\.env\b"),
    ("secret_words", "sensitive_access", 0.35, r"api[_ -]?key|secret|токен|пароль|password|credentials|ключ доступа|private key"),
    ("destructive", "dangerous_command", 0.75, r"rm\s+-rf|mkfs\.|dd\s+if=|drop\s+(table|database)|truncate\s+table|format\s+c:|chmod\s+-r\s+777|:\(\)\s*\{"),
    ("rce_pipe", "dangerous_command", 0.95, r"(curl|wget)\s+[^|]{0,200}\|\s*(ba)?sh|nc\s+-e|bash\s+-i\s*>&|/dev/tcp/|powershell.{0,20}-enc|base64\s+-d\s*\|\s*(ba)?sh|reverse\s*shell|реверс.?шелл"),
    ("exfil", "data_exfiltration", 0.7, r"(отправь|пришли|send|upload|post)\s+.{0,40}(на|to)\s+(http|https|ftp|мою почту|my email|внешн)"),
    ("other_users", "unauthorized_data", 0.8, r"(данные|письма|файлы|переписк\w+|пароли?|data|files|emails?|passwords?)\s+(другого|чужого|всех|других|other|all)\s*(пользовател\w+|users?)?|(of|от)\s+(other|all|других|всех)\s+(users?|пользовател\w+)"),
    ("priv_claim", "privilege_escalation", 0.5, r"(я|i am|i'm)\s+(тут\s+)?(админ\w*|администратор\w*|root|разработчик|admin|developer)|от имени администратора|as (the )?admin|sudo\s+"),
    ("inj_indirect", "indirect_injection", 0.6, r"(assistant|ai|model|ассистент|ии|модель)[,:]?\s+(must|should|need to|обязан\w*|должен|должна|необходимо)\s+(now\s+)?(ignore|send|forward|email|execute|run|delete|игнорир\w+|отправ\w+|выполн\w+|удал\w+)|important (message|instruction)s? (to|for) (the )?(ai|assistant)|важно для (ии|ассистента)"),
    ("jb_hypo", "jailbreak", 0.45, r"(hypothetical(ly)?|гипотетически|в художественных целях|for a novel|для романа|in a fictional world).{0,80}(bypass|ignore|обойти|без ограничений|how to (make|build|hack)|как (взломать|обойти))"),
    ("traversal", "sensitive_access", 0.7, r"(\.\./){2,}|\.\.\\\.\.\\|file:///|/proc/self|/root/\."),
    ("sqli", "dangerous_command", 0.7, r"""['"]\s*or\s+('?1'?\s*=\s*'?1|true)|union\s+(all\s+)?select|;\s*(drop|delete|update|insert)\s|xp_cmdshell|into\s+outfile"""),
    ("ssrf", "data_exfiltration", 0.75, r"169\.254\.169\.254|metadata\.google\.internal"),
    ("secret_leak", "secret_leak", 0.6, r"\bakia[0-9a-z]{16}\b|\bsk-[a-z0-9]{20,}|-----begin [a-z ]*private key-----|ghp_[a-z0-9]{30,}"),
    ("sys_extract_2", "prompt_extraction", 0.8, r"(что|какие)\s+(тебе|у тебя)\s+(написано|сказано|указано)\s+в\s+(систем|начал)|repeat\s+(the\s+)?(text|words)\s+above|повтори\s+(текст|слова)\s+выше|(first|initial)\s+(\d+\s+)?(words|lines)\s+of\s+(your|the)\s+(prompt|instructions)"),
    ("evasion", "evasion", 0.3, r"(decode|декодируй|расшифруй|base64|rot13|hex).{0,40}(and|и)\s+(execute|run|follow|выполни|исполни)"),
    ("tool_abuse", "tool_abuse", 0.4, r"(выполни|запусти|run|execute)\s+.{0,30}(shell|bash|команду|command|terminal|терминал)"),
]
COMPILED = [(i, c, w, re.compile(rx, re.I | re.M)) for i, c, w, rx in RULES]

# Признаки легитимного учебного/технического запроса: снижают вес
BENIGN = re.compile(r"(объясни|расскажи|что (такое|делает)|как работает|в чём разница|explain|what is|what does|how does|documentation|документаци)", re.I)
SOFT_CATS = {"dangerous_command", "sensitive_access", "jailbreak"}
HARD_CATS = {"prompt_injection", "prompt_extraction", "instruction_tampering", "unauthorized_data", "indirect_injection"}


RECO = {"allow": "Передать запрос агенту.", "confirm": "Не передавать до явного подтверждения человеком.",
        "block": "Не передавать агенту; сообщить пользователю об отказе."}


@dataclass
class Result:
    decision: str                # allow | confirm | block
    risk: float
    level: str                   # low | medium | high | critical
    categories: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    matched_rules: list = field(default_factory=list)
    latency_ms: float = 0.0
    recommendation: str = ""


def validate_policy(d: dict):
    if not isinstance(d, dict):
        raise ValueError("Политика должна быть объектом")
    th = d.get("thresholds", {})
    if not (0 <= th.get("confirm", -1) < th.get("block", -1) <= 1):
        raise ValueError("thresholds: нужно 0 <= confirm < block <= 1")
    roles = d.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise ValueError("roles: нужен непустой словарь")
    for n, r in roles.items():
        if not isinstance(r, dict) or not isinstance(r.get("allowed_tools"), list):
            raise ValueError(f"roles.{n}.allowed_tools должен быть списком")
        if not all(isinstance(t, str) and t.strip() for t in r["allowed_tools"]):
            raise ValueError("Имена инструментов должны быть непустыми строками")
    if d.get("default_role") not in roles:
        raise ValueError("default_role отсутствует в roles")
    for k in ("confirm_tools", "forbidden_paths"):
        if not isinstance(d.get(k), list) or not all(isinstance(v, str) and v.strip() for v in d[k]):
            raise ValueError(f"{k} должен быть списком")
    if not isinstance(d.get("own_data_only"), bool):
        raise ValueError("own_data_only должен быть true или false")


class Policy:
    def __init__(self, path="policy.yaml"):
        self.path, self.lock = path, threading.Lock()
        self.reload()

    def reload(self):
        with open(self.path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        validate_policy(d)
        self.data = d

    def save(self, data: dict):
        validate_policy(data)
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp, self.path)
            self.data = data


def _level(r):
    return "critical" if r >= 0.9 else "high" if r >= 0.75 else "medium" if r >= 0.35 else "low"


def check(text: str, policy: Policy, role: Optional[str] = None, user_id: str = "anon",
          tools: Optional[list] = None, judge=None, source: str = "user",
          recent_blocks: int = 0) -> Result:
    t0 = time.perf_counter()
    P = policy.data
    role = role or P["default_role"]
    try:
        args_blob = " ".join(json.dumps(t.get("args", {}), ensure_ascii=False) for t in tools or [])
        views = normalize(text + "\n" + args_blob)  # аргументы инструментов тоже недоверенный ввод
        hits = {}
        for rid, cat, w, rx in COMPILED:
            if any(rx.search(v) for v in views):
                hits[rid] = (cat, w)
        # Only a pure explanation may soften a textual match. Tool calls never do.
        execution = re.search(r"\b(execute|run|send|upload|delete|выполни|исполни|запусти|удали|отправь|пришли)\b", views[0])
        benign = bool(BENIGN.search(views[0])) and not tools and not execution and source == "user"
        # учебный контекст снижает вес «мягких» категорий (не инъекций)
        adj = {rid: (c, w * 0.45 if benign and c in SOFT_CATS else w) for rid, (c, w) in hits.items()}
        prod = 1.0
        for _, w in adj.values():
            prod *= 1 - w
        risk = round(1 - prod, 3)
        cats = sorted({c for c, _ in adj.values()})
        reasons = [f"Сработало правило {rid} ({c})" for rid, (c, _) in adj.items()]
        if benign and hits:
            reasons.append("Обнаружен учебный/технический контекст — вес снижен")
        if len(views) > 7:
            reasons.append("Найден и проверен base64-фрагмент")
        if len(text) > 8000:
            risk = max(risk, 0.4); reasons.append("Подозрительно длинный ввод")

        if source != "user" and set(cats) & HARD_CATS:
            risk = max(risk, 0.8); reasons.append(f"Инструкция внутри недоверенного источника ({source})")
        if recent_blocks >= 2:
            risk = max(min(1.0, risk + 0.1 * min(recent_blocks, 4)), 0.35 if recent_blocks >= 3 else 0)
            reasons.append(f"Повторные нарушения: {recent_blocks} блокировок за 10 минут")

        decision = "allow"
        # --- проверка инструментов и полномочий ---
        role_cfg = P["roles"].get(role)
        if role_cfg is None:
            decision, risk = "block", max(risk, 0.8); reasons.append(f"Неизвестная роль '{role}'")
        else:
            for tool in tools or []:
                name, args = tool.get("name", ""), json.dumps(tool.get("args", {}), ensure_ascii=False).lower()
                if name not in role_cfg["allowed_tools"]:
                    decision, risk = "block", max(risk, 0.85)
                    cats.append("tool_abuse"); reasons.append(f"Инструмент '{name}' не разрешён для роли '{role}'")
                if any(p.lower() in args for p in P["forbidden_paths"]):
                    decision, risk = "block", max(risk, 0.9)
                    cats.append("sensitive_access"); reasons.append(f"Аргументы инструмента '{name}' затрагивают запрещённые пути")
                def owners(value):
                    if isinstance(value, dict):
                        for key, val in value.items():
                            if key in {"user_id", "owner_id"}:
                                yield val
                            yield from owners(val)
                    elif isinstance(value, list):
                        for val in value:
                            yield from owners(val)
                if P.get("own_data_only") and any(owner != user_id for owner in owners(tool.get("args", {}))):
                    decision, risk = "block", max(risk, 0.9)
                    cats.append("unauthorized_data"); reasons.append("Попытка доступа к данным другого пользователя")
                if name in P["confirm_tools"] and decision == "allow":
                    decision = "confirm"; risk = max(risk, P["thresholds"]["confirm"])
                    reasons.append(f"Инструмент '{name}' требует подтверждения по политике")

        # --- серая зона -> LLM-судья (опционально) ---
        th = P["thresholds"]
        if judge and th["confirm"] * 0.5 <= risk < th["block"] and decision != "block":
            j = judge(text)
            if j:
                risk = max(risk, j["risk"]); reasons.append(f"LLM-судья: {j.get('reason', '')}")

        if decision != "block":
            decision = "block" if risk >= th["block"] else "confirm" if risk >= th["confirm"] or decision == "confirm" else "allow"
        return Result(decision, risk, _level(risk), sorted(set(cats)), reasons, list(hits),
                      round((time.perf_counter() - t0) * 1000, 2), RECO[decision])
    except Exception as e:  # fail-safe: при сбое не пропускаем молча
        return Result("confirm", 0.5, "medium", ["internal_error"], [f"Ошибка проверки: {e}"], [],
                      round((time.perf_counter() - t0) * 1000, 2))

