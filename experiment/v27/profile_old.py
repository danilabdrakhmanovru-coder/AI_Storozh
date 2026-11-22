"""Profile cold start and warmed v26 document inference using previously opened diagnostics."""
import cProfile
import io
import os
import pstats
import time
from zipfile import ZipFile
from .data import ROOT,OUT,historical,read
import psutil

def main():
    with ZipFile(OUT/'v26-runtime.zip') as archive:
        if any((ROOT/n).read_bytes()!=archive.read(n) for n in archive.namelist()):
            raise RuntimeError('Run the timing probe on the archived v26 sources')
    os.environ.update(STOROZH_VARIANT='B',STOROZH_ATTACHMENT_INTENT='13',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    proc=psutil.Process()
    proc.cpu_affinity(sorted(proc.cpu_affinity())[-4:])
    from app.contest import ContestRequest, inspect_request, parse_policy, read_policy
    from .evaluate import CTX
    rows=historical()
    raw=read_policy()
    def call(text):
        req=ContestRequest(user_prompt='Summarize the attached document.',context={**CTX,'attachments':[{'content':text}]})
        return inspect_request(req,parse_policy(raw)['thresholds'],raw)
    start=time.monotonic()
    call('The catalogue contains three items.')
    log=['Cold load: %.2f s' % (time.monotonic()-start)]
    for i in (21,375,248):
        profiler=cProfile.Profile()
        profiler.enable()
        verdict=call(rows[i][1])
        profiler.disable()
        buffer=io.StringIO()
        pstats.Stats(profiler,stream=buffer).strip_dirs().sort_stats('cumulative').print_stats(25)
        log+=[str((i,len(rows[i][1]),verdict['decision'])),buffer.getvalue()]
    (OUT/'v26-profile.txt').write_text('\n'.join(log),encoding='utf8')
    print(log[0],flush=True)

if __name__=='__main__':main()
