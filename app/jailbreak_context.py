"""Resolve a lexical jailbreak marker in a document requested for read-only analysis."""
import re

from .guard import normalize

# Recognise a live change of mode close to the marker. Keep standalone mode
# names conservative. Attribution must belong to the same quoted statement.
ACTIVATION = re.compile(
    r'\b(?:enable|activate|enter|switch|start|apply|adopt|use|engage|unlock|turn on|'
    r'включи(?:те)?|активируй(?:те)?|перейди(?:те)?|запусти(?:те)?|используй(?:те)?)\b'
    r'[^.!?;\n]{0,42}$', re.I)
ADDRESSEE = re.compile(r'^\s*(?:yourself|the (?:assistant|agent|model)|себя|ассистента)\b', re.I)
REPORT = re.compile(
    r'(?:\b(?:quoted|recorded|reported|logged|wrote|said|read|displayed)|'
    r'\b(?:цитата|зафиксирован[ао]?|записан[ао]?|написал[аи]?|сообщил[аи]?))'
    r'[^.!?;\n"“«]{0,40}[: ,—-]*$', re.I)
QUOTES = {'"':'"', '“':'”', '«':'»', '‘':'’'}


def attributed(line, start, end):
    for opening, closing in QUOTES.items():
        left = line.rfind(opening, 0, start)
        if left < 0 or not REPORT.search(line[:left]):
            continue
        if closing in line[left+1:start] or line.find(closing, end) < 0:
            continue
        return True
    return False


def reference_candidate(content, marker):
    matches = list(marker.finditer(content))
    if not matches:
        return False
    # An encoded or disguised occurrence is independent evidence. Looking at
    # the masked original also handles a plain mention next to an encoded one.
    rest = marker.sub(' ', content)
    if any(marker.search(view) for view in normalize(rest)):
        return False
    for m in matches:
        begin = content.rfind('\n', 0, m.start()) + 1
        finish = content.find('\n', m.end())
        line = content[begin:finish if finish >= 0 else len(content)]
        start, end = m.start()-begin, m.end()-begin
        prefix, suffix = line[:start], line[end:]
        live = (ACTIVATION.search(prefix) or ADDRESSEE.search(suffix) or
                not (re.search(r'\w',prefix) or re.search(r'\w',suffix)) or
                m.group().casefold() == 'do anything now')
        if live and not attributed(line,start,end):
            return False
    return True
