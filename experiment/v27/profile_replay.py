"""Measure the stages of a previously seen long document; store timings without its contents."""
import cProfile
import io
import os
import pstats
import psutil
import time
from .data import OUT,ROOT,historical
from .evaluate import CTX
from app.contest import ContestRequest,inspect_request,parse_policy
from app.recipient_fast import classify

def main():
    os.environ.update(STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='13')
    proc=psutil.Process();proc.cpu_affinity(sorted(proc.cpu_affinity())[-8:])
    raw=(ROOT/'security_policy.yaml').read_bytes()
    def call(t):
        req=ContestRequest(user_prompt='Summarize the attached document.',context={**CTX,'attachments':[{'content':t}]})
        return inspect_request(req,parse_policy(raw)['thresholds'],raw)
    call('The catalogue has two shelves.')
    text=historical()[410][1]
    name=classify(text)
    report=[name]
    for _ in range(2):
        pr=cProfile.Profile();pr.enable();result=call(text);pr.disable()
        buffer=io.StringIO()
        pstats.Stats(pr,stream=buffer).strip_dirs().sort_stats('cumulative').print_stats(27)
        report.append(result['decision']+'\n'+buffer.getvalue())
    (OUT/'slow-replay.txt').write_text('\n'.join(report),encoding='utf8')
    print(name,result['decision'],flush=True)

if __name__=='__main__':main()
