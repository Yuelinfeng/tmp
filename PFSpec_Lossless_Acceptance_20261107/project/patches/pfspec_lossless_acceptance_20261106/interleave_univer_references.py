"""Run development work between complete formal cohorts and resume the queue."""
import json,os,signal,subprocess,time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
PY='/root/miniconda3/envs/gaussiangpt/bin/python'
PARENT=136910


def children(parent):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[1])==parent and fields[0]!='Z':result.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError,ValueError):continue
    return result


def main():
    metadata=ROOT/'univer_width_interleave_process.json'
    value=dict(pid=os.getpid(),formal_queue_parent=PARENT,status='waiting for current cohort')
    def save():metadata.write_text(json.dumps(value,indent=2)+'\n')
    command=Path(f'/proc/{PARENT}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(ROOT/'source/run_formal_references.sh') not in command:raise RuntimeError('Unexpected formal queue process')
    os.kill(PARENT,signal.SIGSTOP)
    save()
    try:
        while children(PARENT):time.sleep(5)
        value.update(status='running development width references');save()
        code=subprocess.call([PY,str(ROOT/'source/run_univer_width_references.py')])
        value.update(status='complete' if code==0 else 'failed',returncode=code)
    finally:
        try:os.kill(PARENT,signal.SIGCONT)
        except ProcessLookupError:pass
        value['formal_queue_resumed']=True
        value['ended_utc']=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime());save()


if __name__=='__main__':main()
