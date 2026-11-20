"""Learned one-vs-rest jailbreak head for the contest classifier."""
from functools import lru_cache
from pathlib import Path
import re

import joblib

from .intent_features import view

MODEL = Path(__file__).resolve().parents[1] / "experiment/v9/jailbreak.joblib"
# SVM margin selected on the diagnostic sets. This is not a probability.
BLOCK_MARGIN = 0.30
# These are input features. The classifier learns how they interact with
# a request to act, a documentation question and ordinary role-play.
CUES = {
    "persona": r"копи[яию]|персонаж|двойни[кч]|ролев|сыграй|от лица|в образе|вообрази|перевоплот|как будто|за него|представь|вымышлен|imaginary|fictional|alter ego|twin|duplicate|edition|role.play|persona|character|the part|pretend|on behalf|in the voice|speak as",
    "protection": r"безопасн|защит|модерац|огранич|фильтр|предохранител|запрет|safety|safe[- ]?guard|guardrail|moderation|moderated|protecti|restric|constraint|limit|filter|boundar|content polic",
    "removal": r"сним[иать]|снят|выключ|отключ|недейств|приостанов|освобожд|не связан|не действ|не подчин|не соблюд|вне рамок|позволено|без (?:\w+ ){0,2}(?:защит|огранич|фильтр|запрет)|"
               r"exempt|without|unrestrict|unmoderat|uncensor|disabl|suspend|inactive|deactivat|switched off|turn off|set aside|not (?:apply|subject|bound)|no (?:\w+ ){0,2}(?:limit|safety|obligation)|freed|free from|lift|drop|remove|released",
    "inquiry": r"зачем|почему|что (?:такое|значит|делать)|как (?:объяснить|проверить|сообщить)|риск|памятк|отказ|словар|why|what (?:does|is|are|should)|how (?:can|do|should)|refus|declin|risk|glossary|report|training material",
    "awareness": r"распознав|профилактик|обсуждени[яюе]|дискусси|предотвращ|учебн\w* цел|план (?:урока|занятия)|"
                 r"recogniz|awareness|prevention|educational objective|training objective|discussion of|lesson plan",
    "hazard": r"оружи|пистолет|наркот|амфетамин|опиоид|тяж[её]л.{0,12}вред|травм|weapon|firearm|narcotic|illegal drug|injur|physical harm",
    "waive_warning": r"без (?:предупрежден|оговорок)|(?:опусти|пропусти|убери).{0,30}(?:риски|предупрежден|оговорки|меры безопасности)|"
                     r"(?:предупреждени\w*|предостережени\w*) (?:не нужны|опусти)|(?:skip|omit|suppress|without|leave out).{0,30}(?:caveat|warning|safety)",
}
PATTERNS = {name: re.compile(pattern) for name, pattern in CUES.items()}


def model_text(req):
    text = view(req)
    # Compute co-occurrence within a message rather than across a transcript.
    parts = [req.user_prompt, *(m.content for m in req.context.history if m.role in {"user", "assistant"}),
             *(a.content for a in req.context.attachments)]
    tokens = set()
    for part in parts:
        cues = {name for name, pattern in PATTERNS.items() if pattern.search(part.casefold())}
        tokens.update("jbfeature_" + cue for cue in cues)
        if {"hazard", "waive_warning"} <= cues:
            tokens.add("jbfeature_hazard_without_warning")
        if {"protection", "removal"} <= cues:
            tokens.add("jbfeature_lift_protection")
            if "persona" in cues:
                tokens.add("jbfeature_unrestricted_persona")
            if "inquiry" in cues:
                tokens.add("jbfeature_discussion")
    return text + " " + " ".join(sorted(tokens))


@lru_cache(maxsize=1)
def load():
    return joblib.load(MODEL)


def detects(req):
    return bool(load().decision_function([model_text(req)])[0] >= BLOCK_MARGIN)
