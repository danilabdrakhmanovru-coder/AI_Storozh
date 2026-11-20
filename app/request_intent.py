"""CPU classifier for ordinary text, safeguard evasion and other security requests."""
from functools import lru_cache
from pathlib import Path
import re

import joblib

from .jailbreak_intent import features

MODEL = Path(__file__).resolve().parents[1] / "experiment/v12/intent.joblib"
BEHAVIOR = re.compile(
    r"модерир|модерац|проверк.{0,30}(?:ответ|реплик)|защитн.{0,20}проверк|"
    r"фильтр|ограничени|рамки ии|рамок ии|этическ|моральн|цензур|"
    r"screening|safety|safeguard|content filter|restriction|censor|ethic|moral|refus")
SUSPENSION = re.compile(
    r"прекрат|переста|отключ|выключ|отмен|временно сним|приостанов|не (?:примен|действ|связан)|"
    r"нефильтр|аморальн|без огранич|stop|suspend|deactivat|exempt|waiv|unfiltered|amoral|"
    r"unrestricted|no longer|not bound|uncensored|turned off")
ROLE = re.compile(
    r"притвор|персонаж|разыграем|действуй(?:те)? как|отвеча(?:й|йте) как|чат-?бот|"
    r"симулир|сыгра|от имени|persona|act (?:as|like)|role.?play|character|chatbot|pretend|simulate")


def role_change(prompt):
    text = prompt.casefold()
    return bool(ROLE.search(text) and BEHAVIOR.search(text) and SUSPENSION.search(text))


def representation(req):
    prompt = req.user_prompt.casefold()
    flags = []
    if BEHAVIOR.search(prompt) and SUSPENSION.search(prompt):
        flags.append("change")
        if ROLE.search(prompt):
            flags.append("role_change")
    return features(req) + " " + " ".join("jb_request_" + flag for flag in flags)


PERSONA = re.compile(r"персонаж|личност|альтер.эго|отвеча\w* от лица|двойник|в образе|"
                     r"вымышленн|роль|character|persona|alter.ego|roleplay|role.play|fictional")
MANDATE = re.compile(r"всегда (?:да[её]\w*|отвеча\w*)|(?:нельзя|никогда не) отказыва|"
                     r"не след\w* политик|не соблюда\w* правил|нарушая.{0,18}ограничени|"
                     r"never refus|always (?:answer|respond)|must answer every|not follow.{0,16}polic")
OUTPUT = re.compile(r"ответ|реплик|сообщени|response|repl(?:y|ies)|answer|output")
UNCHECKED = re.compile(r"без проверки (?:допустимост|безопасност)|не проход\w* (?:модерац|проверк)|"
                       r"without (?:safety review|content checks)|not subject to moderation")


def current_features(req):
    text = req.user_prompt.casefold()
    flags = []
    if PERSONA.search(text) and MANDATE.search(text):
        flags.append("persona_mandate")
    if OUTPUT.search(text) and UNCHECKED.search(text):
        flags.append("unchecked_output")
    return representation(req) + " " + " ".join("jb_request_" + flag for flag in flags)


@lru_cache(maxsize=1)
def load():
    return joblib.load(MODEL)


def assessment(req):
    model = load()
    scores = dict(zip(model.classes_, model.decision_function([current_features(req)])[0]))
    return scores
