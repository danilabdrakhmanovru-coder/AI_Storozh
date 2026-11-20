"""Semantic features shared by training and CPU inference."""
import base64
import re
import unicodedata
from .guard import HOMOGLYPHS_RU, HOMOGLYPHS, ZERO_WIDTH

# Russian and English stems share sparse features, preserving the original text.
LEXICON = {
    'education': r'объясни|расскажи|зачем|почему|как защит|как распознать|что такое|what is|how to protect|how can we|how do i|explain|чем отлич|как сообщить|how should i|best practice|памятк',
    'abuse': r'оскорб|униж|униз|травл|затрав|травящ|harass|humiliat|insult|abusive|угрожающе|составь угрозу|запуг|racist|racism|threaten|угроз.{0,12}расправ|threat.{0,20}(?:worker|colleague|person)',
    'harm': r'взрыв|отрав|наркот|травм|нападен|оружи|телесн|причин.{0,15}вред|вредонос|кейлоггер|hurt|poison|explosive|weapon|injur|malware|keylogger|ransomware',
    'scam': r'фишинг|фишингов|выман|мошен|фейков|поддельн|обман|phishing|fraud|fake|trick|deceiv|harvest|impersonat',
    'outside': r'внешн|посторонн.{0,14}(?:адрес|получател)|личн.{0,14}(?:сервер|почт)|наружу|за периметр|external|outside|unauthorized recipient|my server',
    'transfer': r'перешли|отправ|вышли|переправ|переда|выгруз|send|upload|forward|export|post to',
    'private': r'чуж|без (?:его |их )?(?:доступа|согласия|разрешения)|другого (?:пациента|клиента|пользователя)|персональн|without (?:their )?(?:permission|consent)|another user|private (?:record|file)|someone else',
    'hidden': r'системн.{0,18}(?:пром|инструкц|сообщен)|скрыт.{0,18}(?:инструкц|указан|настройк)|закрыт.{0,15}инструкц|непубличн|внутренн.{0,12}указан|system (?:message|prompt|instruction)|hidden instruction|startup instruction',
    'extract': r'дослов|процитир|воспроизвед|распечатай|точный текст|оригинал|выведи|скопируй|verbatim|word for word|quote|print|reveal',
    'override': r'замени|перепиши|постоянн.{0,12}правил|нов.{0,12}(?:системн|политик)|поменяй.{0,15}правил|с этого момента|update.{0,15}policy|replace|permanent|override',
    'ignore': r'игнорир|отмени.{0,20}(?:указан|инструкц|огранич)|отбрось|не учитывай|disregard|ignore|discard.{0,15}rule',
    'pretend': r'гипотетич|ролев|персонаж|сыграй|притвор|без (?:цензуры|фильтров|ограничений)|роли.{0,12}помощник|hypothetic|unrestricted|role.play|pretend|developer mode|jailbreak',
    'privilege': r'повысь|повыси|выдай.{0,16}прав|назначь.{0,20}(?:роль|прав)|я (?:админ|владелец)|суперпользоват|администратор|privilege|promot|as admin|treat me as|grant me',
    'secret': r'секрет|парол|приватн.{0,10}ключ|учётн.{0,10}данн|secret|password|private key|credential',
}
PATTERNS = [(key,re.compile(pattern)) for key,pattern in LEXICON.items()]


def features(text):
    text = unicodedata.normalize('NFKC',text).translate(ZERO_WIDTH).casefold()
    def repair(match):
        word=match.group()
        cyr=len(re.findall('[а-яё]',word))
        if 0 < cyr < len(word):
            return word.translate(HOMOGLYPHS_RU if cyr >= len(word)/2 else HOMOGLYPHS)
        return word
    text=re.sub('[a-zа-яё]+',repair,text)
    if re.search(r'задом напер[её]д|read backwards|read in reverse',text) and ':' in text:
        text += ' '+text.partition(':')[2][::-1]
    words = [key for key,pattern in PATTERNS if pattern.search(text)]
    return text + ' ' + ' '.join('intent_'+w for w in words * 2)


def view(req):
    def clean(s):
        extra=[]
        for token in re.findall(r'[A-Za-z0-9+/=]{16,}',s)[:8]:
            try:
                dec=base64.b64decode(token,validate=True).decode()
                if dec.isprintable(): extra.append(dec)
            except (ValueError,UnicodeError): pass
        return features(s+' '+' '.join(extra))
    # The summary instruction is often identical for an attack and a benign document.
    # Put most of the feature weight on what the document actually contains.
    parts=[clean(req.user_prompt)]
    for m in req.context.history:
        if m.role in ('user','assistant'): parts.append('history '+clean(m.content))
    for a in req.context.attachments: parts.extend([clean(a.content)]*2)
    return '\n'.join(parts)

