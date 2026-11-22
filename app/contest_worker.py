"""Serialized inference with a warmed spare. Each packet owns exactly one pipe reply."""
import atexit
from collections import deque
import multiprocessing
import os
import threading
import time

BUDGET_SECONDS=2.65

class BudgetExceeded(TimeoutError):
    pass

def serve(conn):
    from .contest import ContestRequest,inspect_request,parse_policy,read_policy
    from .category_model import load_current,load_severity
    from .classifier import load_classifier
    from .jailbreak_intent import load as jailbreak_model
    from .request_intent import load as intent_model
    try:
        # A benign probe alone need not traverse every category-dependent branch.
        for load in (load_classifier,load_current,load_severity,jailbreak_model,intent_model):
            load()
        if os.getenv('STOROZH_ATTACHMENT_INTENT')=='8' and os.getenv('STOROZH_VARIANT','B')=='B':
            from .semantic_review import reviewer
            reviewer().assess('Summarize the inventory.')
        if os.getenv('STOROZH_ATTACHMENT_INTENT')=='9' and os.getenv('STOROZH_VARIANT','B')=='B':
            from .intent_review import reviewer
            reviewer().assess('Summarize the inventory.')
        if os.getenv('STOROZH_ATTACHMENT_INTENT') in ('10','11','12','13') and os.getenv('STOROZH_VARIANT','B')=='B':
            from .confidence_review import reviewer
            reviewer()
        raw=read_policy()
        policy=parse_policy(raw)
        agent=next(iter(policy['agents']))
        role=next(iter(policy['agents'][agent]['roles']))
        for content in ('Notes from the inventory.', 'Inventory register.\n'*80):
            dummy=ContestRequest(user_prompt='Summarize the inventory.',context={
                'target_agent':agent,'user_role':role,'attachments':[{'content':content}]})
            inspect_request(dummy,policy['thresholds'],raw)
        if os.getenv('STOROZH_ATTACHMENT_INTENT')=='13' and os.getenv('STOROZH_VARIANT','B')=='B':
            from . import attachment_cascade, attachment_passage, document_encoder, document_review
            from .recipient_calibration import classify
            for load in (attachment_cascade.model,attachment_cascade.local_svm,attachment_passage.load,
                         document_encoder.load,document_review.model):
                load()
            classify('The catalogue contains three items.')
        conn.send('ready')
        while True:
            packet,limits,policy=conn.recv()
            try:
                value=inspect_request(ContestRequest.model_validate(packet),limits,policy)
                conn.send((True,value))
            except Exception:
                conn.send((False,None))
    except (EOFError,OSError):
        pass
    finally:
        conn.close()

def discard(slot):
    proc,conn=slot
    conn.close()
    if proc.is_alive():
        proc.kill()
    proc.join(timeout=.2)


class GuardWorker:
    startup_seconds=120

    def __init__(self,target=serve):
        self.target=target
        self.request_lock=threading.Lock()
        self.condition=threading.Condition()
        self.ready=deque()
        self.busy=False
        self.builder=None
        self.stop=threading.Event()
        self.failure=None

    @property
    def proc(self):
        with self.condition:
            return self.ready[0][0] if self.ready else None

    def _spawn(self,stop):
        ctx=multiprocessing.get_context('spawn')
        parent,child=ctx.Pipe()
        proc=ctx.Process(target=self.target,args=(child,),daemon=True)
        try:
            proc.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        slot=proc,parent
        deadline=time.monotonic()+self.startup_seconds
        try:
            while not stop.is_set() and time.monotonic()<deadline:
                if parent.poll(.1):
                    if parent.recv()=='ready':
                        return slot
                    raise RuntimeError('Некорректный ответ при загрузке модели')
            raise TimeoutError('Прогрев модели не завершён')
        except BaseException:
            discard(slot)
            raise

    def _fill(self):
        try:
            while True:
                with self.condition:
                    if self.stop.is_set() or len(self.ready)+self.busy>=2:
                        return
                slot=self._spawn(self.stop)
                with self.condition:
                    if self.stop.is_set():
                        discard(slot)
                        return
                    self.ready.append(slot)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.failure=exc
        finally:
            with self.condition:
                self.builder=None
                self.condition.notify_all()

    def _start_fill(self):
        # Caller holds condition. One builder owns all cold starts.
        if self.builder is None and len(self.ready)+self.busy<2 and not self.stop.is_set():
            self.failure=None
            self.builder=threading.Thread(target=self._fill,daemon=True)
            self.builder.start()

    def _wait(self,count,deadline):
        self._start_fill()
        while len(self.ready)<count:
            if self.stop.is_set() or self.failure:
                raise RuntimeError('Модель недоступна') from self.failure
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise BudgetExceeded()
            # A builder can exit just as a slot is retired.
            self._start_fill()
            self.condition.wait(remaining)

    def warm(self):
        with self.request_lock,self.condition:
            self.stop.clear()
            self._wait(2,time.monotonic()+2*self.startup_seconds+2)

    def check(self,packet,limits,policy,deadline):
        if not self.request_lock.acquire(timeout=max(0.,deadline-time.monotonic())):
            raise BudgetExceeded()
        try:
            with self.condition:
                self._wait(1,deadline)
                if time.monotonic()>=deadline:
                    raise BudgetExceeded()
                slot=self.ready.popleft()
                self.busy=True
            reusable=False
            try:
                conn=slot[1]
                conn.send((packet,limits,policy))
                if not conn.poll(max(0.,deadline-time.monotonic())):
                    raise BudgetExceeded()
                ok,result=conn.recv()
                if time.monotonic()>deadline:
                    raise BudgetExceeded()
                reusable=True
                if not ok:
                    raise RuntimeError('Модель не завершила проверку')
                return result
            finally:
                if not reusable:
                    discard(slot)
                with self.condition:
                    self.busy=False
                    if reusable:
                        self.ready.appendleft(slot)
                    self._start_fill()
                    self.condition.notify_all()
        finally:
            self.request_lock.release()

    def close(self):
        with self.request_lock:
            with self.condition:
                self.stop.set()
                builder=self.builder
                self.condition.notify_all()
            if builder:
                builder.join()
            while self.ready:
                discard(self.ready.pop())

worker=GuardWorker()
atexit.register(worker.close)

