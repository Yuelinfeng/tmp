"""Retain completion or failure for a detached research script."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    args=parser.parse_args()
    code=subprocess.call(['bash',str(args.script)])
    value=json.loads(args.metadata.read_text())
    value.update(status='complete' if code==0 else 'failed',returncode=code,
                 ended_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()))
    args.metadata.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':
    main()
