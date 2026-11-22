"""Validated local policy snapshots. A bad edit cannot replace a working snapshot."""
import hashlib
import os
from functools import lru_cache
from pathlib import Path
import re
import threading
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
LEGACY = yaml.safe_load(Path(__file__).with_name("legacy_policy_defaults.yaml").read_bytes())
LOCK = threading.RLock()
snapshots = {}


def string_list(value):
    if not isinstance(value, list) or any(not isinstance(s, str) or not s.strip() for s in value):
        raise ValueError("Ожидается список непустых строк.")
    return value


def pattern_list(values):
    for pattern in string_list(values):
        if len(pattern) > 2048:
            raise ValueError("Слишком длинное регулярное выражение.")
        re.compile(pattern, re.I)
    return values


@lru_cache(maxsize=16)
def parse_policy(raw):
    if len(raw) > 262144:
        raise ValueError("Размер политики превышает 256 КиБ.")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or not isinstance(data.get("agents"),dict) or not data["agents"]:
        raise ValueError("Нужна непустая секция agents.")
    for key in ("always_block","context_dependent"):
        string_list(data[key])
    if set(data["always_block"]) & set(data["context_dependent"]):
        raise ValueError("Категория включена в две группы.")
    limits = data.setdefault("thresholds", {"confirm":.35,"block":.75})
    if not 0 <= limits["confirm"] <= limits["block"] <= 1:
        raise ValueError("Некорректные пороги.")
    for agent in data["agents"].values():
        names = string_list(agent["tools"])
        envs = agent["environments"]
        if not envs or not agent["roles"] or set(envs) & {"allow","confirm","block","any","deny"}:
            raise ValueError("Проверьте имена окружений и роли.")
        for globs in envs.values():
            string_list(globs)
        specs = agent.setdefault("tool_specs", {})
        if set(specs) - set(names):
            raise ValueError("Описание ссылается на отсутствующий инструмент.")
        for name in names:
            spec = specs.setdefault(name, dict(LEGACY["tool_specs"].get(name, {"type":"execute","aliases":[]})))
            if spec["type"] not in ("read","modify","destroy","execute"):
                raise ValueError("Неизвестный тип действия.")
            aliases = [name,*string_list(spec.get("aliases",[]))]
            patterns = [*pattern_list(spec.get("patterns",[])),*(r"(?<!\w)"+re.escape(s)+r"(?!\w)" for s in aliases)]
            spec["_match"] = re.compile("|".join("(?:"+p+")" for p in patterns),re.I)
            spec["_literal"] = [re.compile(p,re.I) for p in pattern_list(spec.get("literal_patterns",[]))]
        agent.setdefault("critical_environments",[s for s in LEGACY["critical_environments"] if s in envs])
        agent.setdefault("environment_aliases",{e:v for e,v in LEGACY["environment_aliases"].items() if e in envs})
        if set(string_list(agent["critical_environments"])) - set(envs) or set(agent["environment_aliases"]) - set(envs):
            raise ValueError("Неизвестное окружение.")
        for aliases in agent["environment_aliases"].values():
            string_list(aliases)
        for role in agent["roles"].values():
            if set(role["tools"]) - set(names):
                raise ValueError("Право ссылается на отсутствующий инструмент.")
            for rule in role["tools"].values():
                if isinstance(rule,dict):
                    if set(rule) - {"any",*envs} or set(rule.values()) - {"allow","deny","confirm","block","any"}:
                        raise ValueError("Некорректное правило по окружению.")
                elif rule not in {"allow","confirm","deny","block","any",*envs}:
                    raise ValueError("Неизвестное право.")
        if type(agent.get('review_correspondence',False)) is not bool:
            raise ValueError('review_correspondence должно быть логическим значением')
        legacy_tools=set(names) & set(LEGACY['tool_specs'])
        pattern_list(agent.setdefault('definition_patterns',LEGACY['definition_patterns'] if legacy_tools else []))
        for item in agent.setdefault('protected_data',LEGACY['protected_data'] if any('secrets' in r for r in agent['roles'].values()) else []):
            if not isinstance(item['permission'],str):raise ValueError("Требуется имя разрешения.")
            pattern_list([item[k] for k in ('pattern','metadata_pattern','metadata_literal')])
        for spec in specs.values():
            pattern_list([spec[k] for k in ('inline_literal','inline_repeated') if k in spec])
            if 'inline_repeated' in spec and re.compile(spec['inline_repeated']).groups!=1:raise ValueError('Для повторов нужна одна группа.')
        rules = []
        for item in agent.get("forbidden_actions",[]):
            rule = LEGACY["forbidden_actions"][item] if isinstance(item,str) else item
            if not rule["name"] or not rule["description"] or not pattern_list(rule["patterns"]):
                raise ValueError("У запрета должны быть имя, описание и шаблоны.")
            rules.append(rule)
        agent["forbidden_actions"] = rules
    return data


def path():
    return Path(os.getenv("ASTRA_POLICY_PATH", ROOT/"security_policy.yaml")).resolve()


def read_policy(strict=False):
    target = path()
    with LOCK:
        try:
            raw = target.read_bytes()
            parse_policy(raw)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, re.error, yaml.YAMLError):
            if strict or target not in snapshots:
                raise
            return snapshots[target]
        snapshots[target] = raw
        return raw


def apply_policy(raw, expected):
    target = path()
    parse_policy(raw)
    with LOCK:
        if hashlib.sha256(read_policy()).hexdigest() != expected:
            raise ValueError("Версия изменилась. Обновите страницу.")
        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as file:
            tmp = Path(file.name)
            file.write(raw)
        try:
            os.replace(tmp,target)
        finally:
            tmp.unlink(missing_ok=True)
        snapshots[target] = raw

