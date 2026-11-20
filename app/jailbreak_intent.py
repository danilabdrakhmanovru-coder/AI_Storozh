"""Features and calibrated classifier for direct requests to lift model safeguards."""
from functools import lru_cache
from pathlib import Path
import re

import joblib

from .jailbreak_model import model_text, PATTERNS

MODEL = Path(__file__).resolve().parents[1] / "experiment/v10/jailbreak.joblib"
BLOCK_MARGIN = 0.30
CONTROL = re.compile(r"модерир|модераци|провер.{0,20}безопасност|огранич|защит|фильтр|цензур|"
                     r"moderat|safety|content (?:screen|check|protect|review)|safe[- ]?guard|"
                     r"guardrail|restrictions?|limits?|boundar|uncensor|unfiltered|amoral")
EXCEPTION = re.compile(r"приостан|отмен|разреша|разрешил|позвол|вправе|необязательн|перестан|прекрат|"
                       r"обход|не примен|не (?:распространя|действ)|за предел|вне действ|"
                       r"authoriz|permission|excus|except|grant|waiv|stop|cease|outside|optional|"
                       r"on (?:hold|pause)|paused|turned off|out of (?:effect|service)|no longer|not in force|"
                       r"inactive|off[- ]?line|not applicable|suspend|dispense")
RESPONSE = re.compile(r"реплик|ответ|сообщени|собственн|ассистент|помощник|чат-?бот|"
                      r"output|repl(?:y|ies)|response|answer|assistant|bot|character")
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
        if PATTERNS["persona"].search(t):
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
    return load().decision_function([features(req)])[0] >= BLOCK_MARGIN


def may_reclassify(req):
    # A new label for an already dangerous request needs evidence in the
    # current instruction. Keep the original source handling for attachments.
    return not req.context.attachments and bool(signals(req.user_prompt))
