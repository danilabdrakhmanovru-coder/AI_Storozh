"""CPU classifier for attempts by attached text to control the reader's answer."""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np

MODEL = Path(__file__).with_name('weights') / 'attachment-intent.joblib'
WINDOW = 1000
OVERLAP = 280

# Features describe who is addressed, what output is being changed and whether
# the sentence recounts an earlier exchange. None depends on roles or tools.
CONCEPTS = {
    'recipient': r'\b(?:пользовател|заявител|запрашивающ|автору (?:запроса|вопроса)|the user|the requester|the person who)',
    'machine': r'\b(?:ассистент|помощник|модел[ьи]|чат.?бот|бот\b|ии\b|assistant|summarizer|bot\b|language model|reading system)',
    'own': r'\b(?:сво[йёеия]|тво[йёеия]|в сво[её]м|your (?:own |final |entire |next |actual )?(?:answer|response|reply|findings)|you (?:are|send|respond|prepare|reply))',
    'output': r'\b(?:ответ|отвеч|итог|резюм|вывод|пересказ|сводк|отч[её]т|заключен|сообщени|reply|repl(?:ies|ying)|answer|output|response|summary|digest|findings|report|analysis|conclusion)',
    'replace': r'\b(?:вместо|замен|заменя|подмен|подстав|независимо|вне зависимости|какой бы|instead|in (?:its |the )?place|replace|substitut|rather than|regardless|whatever)',
    'restrict': r'\b(?:только|одн[уио]|единствен|исключительн|ограничься|ничего|не (?:включай|сообщай|приводи|упоминай|пересказывай)|пропуст|скрой|умолчи|опусти|only|just|single|alone|nothing else|omit|suppress|skip|hide|do not|leave out)',
    'directive': r'\b(?:поставь|сделай|выведи|верни|вставь|добавь|сообщи|выдай|укажи|напиши|ответь|скажи|напечатай|передай|замени|подмени|подставь|отправь|покажи|заверши|оставь|включи|повтори|ограничься|return|write|reply|respond|output|print|give|send|add|append|replace|make|provide|use|insert|show|report|declare|certify|state|tell|say|emit|substitute)\b',
    'required': r'\b(?:обязан|должен|должна|должно|пусть|требуется|must|shall|should|needs? to)',
    'review': r'\b(?:согласован|одобрен|проверка пройдена|проверк|замечан|допуск|approved|approval|passed|sign.off|safety|inspection|assessment)',
    'frame': r'\b(?:пример|образец|учебн|учител|педагог|урок|преподавател|цитат|цитир|реплик|стенограм|протокол|иллюстр|дословн|ошибк|ошибочн|вчера|ранее|прошл|записан|зафиксир|разобра|обсужд|разбор|упражнен|example|sample|quot|transcript|lesson|previous|earlier|yesterday|training|exercise|recorded|documented|discuss|reported|witness|recalled|incorrect|under review)',
    'human': r'\b(?:участник|игрок|читательск|сотрудник|бланк|карточк|поле|бумаг|бумажн|распечат|куратор|секретар|лаборант|диспетчер|редактор|таблиц|надпис|афиш|этикетк|конверт|стенд|макет|билет|анкете|student|participant|player|clerk|collector|editor|librarian|curator|technician|volunteer|colleague|visitor|paper|printed|worksheet|ticket|envelope|postcard|label|ballot|form|exhibit|caption)',
    'past': r'\b(?:ответил|указал|попросил|сообщил|заменил|скрыл|записал|прив[её]л|приведен|содержал|было|returned|replied|answered|replaced|stated|said|was|were|contained|asked|produced|gave|printed)\w*',
}
PATTERNS = {name:re.compile(pattern,re.I) for name,pattern in CONCEPTS.items()}
QUOTES = re.compile(r'«([^»]{8,700})»|“([^”]{8,700})”|"([^"\n]{8,700})"')
REPORTED_QUOTE = re.compile(r'цит|реплик|учебн|учител|стенограмм|протокол|пример|разбор|упражнен|ошибочн|прив[еёо]д|записан|произн[её]с|обсужд|комментари|quote|example|transcript|exercise|discuss|recorded|the sentence|the phrase|demonstration|the lesson|recalled|faulty',re.I)
REPORTED_COLON = re.compile(r'([^.!?\n:]{4,160}):\s*([^.!?\n]{3,350})')
REPORT_VERB = re.compile(r'привод|привед|прив[её]л|цитир|записа|процитир|показан|звучал|разобра|illustrat|quotes?|reads?|transcrib|recorded|discuss|examples?|упражнени|реплик|стенограмм|пример\s*$|цитата\s*$',re.I)
PHYSICAL = re.compile(
    r'бумаж|типограф|на (?:обороте|стенде|бланке|полях|конверте)|карточк|анкет|билет|граф[аеы]|'
    r'печа(?:тн|ти)|макет|экспонат|тираж|викторин|конкурс|игр[ауыое]|игрок|ученик|участник|'
    r'конверт|открытк|витрин|пьес|реквизит|кассир|художник|коллег[ае]|режисс[её]р|'
    r'\b(?:paper|printed|postcard|postage|envelope|worksheet|coupon|voucher|ticket|ballot|'
    r'quiz|contestant|player|performer|actor|cashier|curator|librarian|clerk|typesetter|'
    r'booth|exhibition|caption|recipient of the letter|answer sheet|receipt|tear.off|stage manager)\b',re.I)
DIRECT_READER = re.compile(
    r'(?:ассистент|чат.?бот|помощник|модель|бот)\s*[,!]|(?:ассистент|модел[ьи]|бот)\W+(?:читаю|которая|обрабатыва)|'
    r'\b(?:assistant|bot|model|summarizer)\s*(?:[,!]|processing\b|handling\b|reading\b|reviewing\b|drafting\b)|'
    r'\b(?:to the (?:assistant|summarizer|model)|читающ\w* (?:бот|модел|ассистент))',re.I)
CONTENT_SUBSTITUTION = re.compile(
    r'(?:вместо|подмен|замен|пропуст|опусти|скрой|отбрось|не (?:упоминай|пересказывай|перечисляй)).{0,55}'
    r'(?:пересказ|резюм|факт|вывод|запрошенн|сведени|обзор|содержани|запрос|вопрос|итог|отч[её]т|ответ)|'
    r'(?:instead of|in (?:lieu|place) of|skip|omit|hide|suppress|replace|do not (?:tell|report|summarize)).{0,70}'
    r'(?:findings|requested|the (?:conclusions|actual|evidence|review|summary|content)|your (?:response|answer)|analysis|what the)',re.I)


def attributed_text(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    # Attribute cited speech to its speaker. A live instruction outside
    # the citation remains visible to both feature channels.
    def attributed(match):
        around=text[max(0,match.start()-130):match.start()]+text[match.end():match.end()+65]
        speech_label=re.search(r'(?:диалог|пьес|драматург|говорит|расшифров|dialogue|playwright|character says)',around,re.I)
        return ' quoted_example. ' if REPORTED_QUOTE.search(around) or speech_label else match.group()
    voice=QUOTES.sub(attributed,text)
    def reported(match):
        lead=match[1]
        if REPORTED_QUOTE.search(lead) and REPORT_VERB.search(lead):
            return lead+': quoted_example '
        return match[0]
    return REPORTED_COLON.sub(reported,voice)


def discourse(text):
    voice=attributed_text(text)
    values={k:bool(expr.search(voice)) for k,expr in PATTERNS.items()}
    order=bool(re.search(r'\b(?:ответь|выведи|верни|напиши|напечатай|сообщи|покажи|добавь|поставь|оставь|выдай|укажи|замени|скрой|пропусти|подмени|подставь|отправь|повтори|заверши|опусти|объяви|отрази|отметь|подтверди|сделай|впиши|вставь|передай|ограничься|пусть|прекрати|не пересказывай|не перечисляй|не выдавай)\b', voice))
    # English infinitives can also name an operation. Sentence position or
    # an explicit obligation helps distinguish the imperative.
    english=bool(re.search(r'(?:^|[.!?:;\n,]\s*|\band\s+)(?:please\s+)?(?:do not|only\s+)?(?:reply|respond|return|answer|give|write|print|output|send|show|make|report|omit|skip|append|replace|substitute|certify|tell|state|use|hide|declare|add|emit|leave)\b', voice))
    values['command']=order or english or values['required']
    values['redirect']=values['replace'] or values['restrict']
    values['audience']=values['own'] or values['machine'] or values['recipient']
    values['output_order']=values['output'] and values['command']
    values['directed_change']=values['output_order'] and (values['audience'] or values['redirect'])
    values['attributed_quote']='quoted_example' in voice
    values['physical']=bool(PHYSICAL.search(voice))
    values['direct_reader']=bool(DIRECT_READER.search(voice))
    values['substitute_content']=bool(CONTENT_SUBSTITUTION.search(voice))
    values['human_task']=values['physical'] and not values['direct_reader'] and not values['substitute_content']
    values['fabricated_status']=values['review'] and values['command'] and bool(re.search(
        r'несмотря|хотя|независимо|без уч[её]та|вместо|положительн|одобрен|despite|even (?:if|though)|regardless|instead|as (?:passed|approved)|without regard',voice))
    positive=[k for k,v in values.items() if v]
    return {key:1.0 for key in positive+[f'{a}+{b}' for i,a in enumerate(positive) for b in positive[i+1:]]}


def discourse_batch(texts):
    return [discourse(t) for t in texts]


def scores_for(classifier,texts):
    """Ambiguous human paperwork falls back to the existing gateway checks."""
    scores=np.full(len(texts),-np.inf)
    eligible=[]
    for i,text in enumerate(texts):
        current=discourse(text)
        if 'command' not in current:
            continue
        if 'human_task' in current:
            # A genuine addressee change in a later sentence should still
            # be assessed, even when the preceding notes concern a paper form.
            clauses=re.split(r'[.!?\n]+',attributed_text(text))
            directed=False
            for clause in clauses:
                if PHYSICAL.search(clause):
                    continue
                signals=discourse(clause)
                if {'command','recipient','output','redirect'} <= signals.keys():
                    directed=True
                    break
            if not directed:
                continue
        eligible.append(i)
    if eligible:
        scores[eligible]=classifier.decision_function([texts[i] for i in eligible])
    return scores


def windows(text):
    if not text.strip():
        return
    start=0
    while start<len(text):
        end=min(len(text),start+WINDOW)
        yield start,end,text[start:end]
        if end==len(text):break
        start=end-OVERLAP


@lru_cache(maxsize=1)
def load():
    return joblib.load(MODEL)


def detect(attachments):
    spans=[(index,start,end,window) for index,attachment in enumerate(attachments)
           for start,end,window in windows(attachment.content)]
    if not spans:return None
    model=load()
    scores=scores_for(model['classifier'],[s[3] for s in spans])
    best=int(scores.argmax())
    if scores[best]<model['threshold']:return None
    index,start,end,_=spans[best]
    return dict(source='attachment',index=index,start=start,end=end,rule='attachment_intent')

