"""CPU classifier for ordinary text, safeguard evasion and other security requests."""
from functools import lru_cache
from pathlib import Path
import re

import joblib

from .jailbreak_intent import features

MODEL = Path(__file__).resolve().parents[1] / "experiment/v13/intent.joblib"
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


CHECK = re.compile(r"проверк\w* допустимост|провер\w*.{0,24}безопасност|защит\w*.{0,30}провер|"
                   r"safety (?:review|screening|checks?)|content review|output screening")
LIFT = re.compile(r"отмен[иая]|переста|приостанов|не проверял|больше не|без проверк|не применя|"
                  r"не проход|on hold|suspend|dispense|stop|switch off|set aside|cease")
OUTPUT_SCOPE = re.compile(r"отвеча|ответ|реплик|сообщени|выда[её]|repl(?:y|ies)|outputs?|responses?|answers?")
UNBOUND = re.compile(r"не связан|нет ограничен|без.{0,15}(?:морал|ограничен)|безусловно допустим|"
                     r"люб\w+ (?:просьб|запрос)|вс[её] (?:разрешено|допустимо)|"
                     r"no (?:ethics|limits|restrictions)|not bound|anything (?:goes|is allowed)|"
                     r"unconditional compliance|unrestricted|amoral|never say no")

# Keep a request to change the assistant's own output checks in one clause.
# This feature is deliberately absent from questions, quotations and negations.
REQUEST_REVIEW_CHANGE = re.compile(
    r'(?:^|[,;]\s*)(?:(?:please|пожалуйста|я прошу)\s+)?'
    r'(?:отмен(?:и|ить)|считай|отложи|прекрати|останови|переведи|поставь|'
    r'set|put|treat|consider|suspend|pause|leave|stop|disable|lift)\b')
REVIEW_OWNER = re.compile(r'\b(?:тво\w*|сво\w*|собственн\w*|ты будешь|ты отвеча\w*|'
                          r'your|outgoing|assistant.s|the bot.s)\b')
REVIEW_OBJECT = re.compile(r'допустимост|модерац|проверк\w* (?:на )?безопасност|защитн\w* проверк|'
                           r'safeguards?|safety (?:checks?|review|screening)|'
                           r'(?:screen|controls? on|moderation of).{0,24}(?:repl|responses?|output)')
INACTIVE = re.compile(r'приостанов|отмен|на пауз|необязательн|недейств|прекрат|останов|отлож|'
                      r'inactive|on hold|on pause|suspend|aside|revok|disabled?|no longer|stop|lift')
DISCUSSION = re.compile(r'["«»“”\n?]|\b(?:не|not|never|why|explain|whether|translate|discuss why|'
                        r'почему|зачем|объясни|переведи фразу|отказ|обсуди|цитат\w*)\b')


def direct_suspension(prompt):
    if len(prompt) > 340:
        return False
    text = prompt.casefold()
    if DISCUSSION.search(text):
        return False
    match = REQUEST_REVIEW_CHANGE.search(text)
    if not match:
        return False
    clause = text[match.start():].split('.', 1)[0]
    output_target = OUTPUT_SCOPE.search(clause) or re.search(r'что ты (?:будешь|говоришь)|what you say', clause)
    return bool(output_target and REVIEW_OWNER.search(clause) and REVIEW_OBJECT.search(clause) and INACTIVE.search(clause))


def scoped_features(req):
    text = req.user_prompt.casefold()
    flags = []
    if CHECK.search(text) and LIFT.search(text) and OUTPUT_SCOPE.search(text):
        flags.append("suspend_output_review")
    if (ROLE.search(text) or PERSONA.search(text)) and UNBOUND.search(text):
        flags.append("unbound_speaker")
    if direct_suspension(req.user_prompt):
        flags.append("direct_suspension")
    return current_features(req) + " " + " ".join("jb_request_" + flag for flag in flags)


@lru_cache(maxsize=1)
def load():
    return joblib.load(MODEL)


def assessment(req):
    model = load()
    scores = dict(zip(model.classes_, model.decision_function([scoped_features(req)])[0]))
    text = req.user_prompt
    if (len(text) >= 600 and not req.context.attachments and not req.context.history
            and -0.55 < scores['jailbreak'] < 0.60 and (ROLE.search(text.casefold()) or PERSONA.search(text.casefold()))):
        # Long role descriptions may contain a single paragraph defining the
        # requested behaviour. Only a strongly positive local score counts.
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        windows = []
        for i in range(len(sentences)):
            segment = " ".join(sentences[max(0, i-1):i+2])[:1200]
            if (UNBOUND.search(sentences[i].casefold()) or MANDATE.search(sentences[i].casefold())):
                windows.append(scoped_features(req.model_copy(update={'user_prompt': segment})))
                if len(windows) == 4:
                    break
        if windows:
            margins = model.decision_function(windows)
            column = list(model.classes_).index('jailbreak')
            local = margins[margins[:, column].argmax()]
            if local[column] >= 0.70:
                return dict(zip(model.classes_, local))
    return scores
