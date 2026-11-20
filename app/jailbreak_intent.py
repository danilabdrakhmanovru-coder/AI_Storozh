"""Features and calibrated classifier for direct requests to lift model safeguards."""
from functools import lru_cache
from pathlib import Path
import re

import joblib

from .jailbreak_model import model_text, PATTERNS

MODEL = Path(__file__).resolve().parents[1] / "experiment/v10/jailbreak.joblib"
DIRECT_MARGIN = 0.20
BROAD_MARGIN = 0.65
ROLE_MARGIN = 0.05
CONTROL = re.compile(r"модерир|модераци|провер.{0,32}(?:безопасност|реплик|ответ)|морал|этическ|огранич|защит|фильтр|цензур|"
                     r"moderat|safety|content (?:screen|check|protect|review|control)|protective checks|safe[- ]?guard|"
                     r"guardrail|restrictions?|limits?|boundar|uncensor|unfiltered|amoral|ethical|moral code")
EXCEPTION = re.compile(r"приостан|отмен|разреша|разрешил|позвол|вправе|необязательн|перестан|прекрат|"
                       r"обход|неактив|выведен|освобож[д]|не примен|не проход|не (?:распространя|действ)|за предел|вне действ|"
                       r"authoriz|permission|excus|except|grant|waiv|stop|cease|outside|optional|"
                       r"on (?:hold|pause)|paused|turned off|out of (?:effect|service)|no longer|not in force|"
                       r"inactive|off[- ]?line|not applicable|suspend|dispense|deactivat|unfiltered|amoral|uncensor|"
                       r"not bound|free of|no (?:ethic|moral|restrict)|без (?:морал|этическ|цензур)")
RESPONSE = re.compile(r"реплик|ответ|сообщени|собственн|ассистент|помощник|чат-?бот|"
                      r"output|repl(?:y|ies)|response|answer|assistant|bot|character")
PERSONA = re.compile(r"ты (?:теперь|будешь)|отныне ты|предстоит играть|войти в роль|отвеча[йть]+ как|"
                     r"act as|you are (?:now|a |an )|you will (?:act|play)|play the role|another (?:ai|model)|"
                     r"named|persona|chatbot|digital twin|pretend to be")
SUPPRESS = re.compile(r"(?:не (?:включая|добавляя|упоминая)|не нужны|никаких|опуст\w*|без)\s*.{0,25}"
                      r"(?:предупрежден|оговорок|предостережен)|(?:leave|omit|skip|no|without).{0,35}"
                      r"(?:warnings?|caveats?|safety considerations?)")


def signals(text):
    t = text.casefold()
    lift = EXCEPTION.search(t) or PATTERNS["removal"].search(t)
    flags = set()
    if CONTROL.search(t) and lift:
        flags.add("lift")
        if RESPONSE.search(t):
            flags.add("reply")
        if PATTERNS["persona"].search(t) or PERSONA.search(t):
            flags.add("role")
        if PATTERNS["inquiry"].search(t) or PATTERNS["awareness"].search(t):
            flags.add("discuss")
    if PATTERNS["hazard"].search(t) and (SUPPRESS.search(t) or PATTERNS["waive_warning"].search(t)):
        flags.add("omit_warning")
    return flags


def features(req):
    flags = signals(req.user_prompt)
    return model_text(req) + " " + " ".join("jb_request_" + key for key in sorted(flags))


@lru_cache(maxsize=1)
def load():
    return joblib.load(MODEL)


def detects(req):
    flags = signals(req.user_prompt)
    margin = BROAD_MARGIN
    if flags & {"reply", "role", "omit_warning"}:
        margin = DIRECT_MARGIN
    if {"role", "reply"} <= flags and "discuss" not in flags:
        margin = ROLE_MARGIN
    model = load()
    if model.decision_function([features(req)])[0] >= margin:
        return True
    # A long persona description can bury a short direct change of safety
    # behavior. Score bounded, adjacent sentences only with local evidence.
    if req.context.attachments or len(req.user_prompt) < 600 or "role" not in flags:
        return False
    sentences = re.split(r"(?<=[.!?])\s+|\n+", req.user_prompt)
    windows = []
    for i in range(len(sentences)):
        text = " ".join(sentences[i:i+2])[:900]
        evidence = signals(text)
        if {"role", "reply"} <= evidence and "discuss" not in evidence:
            excerpt = req.model_copy(update={"user_prompt": text})
            windows.append(features(excerpt))
            if len(windows) == 4:
                break
    return bool(windows) and max(model.decision_function(windows)) >= BROAD_MARGIN


def may_reclassify(req):
    # A new label for an already dangerous request needs evidence in the
    # current instruction. Keep the original source handling for attachments.
    return not req.context.attachments and bool(signals(req.user_prompt) & {"reply", "role", "omit_warning"})
