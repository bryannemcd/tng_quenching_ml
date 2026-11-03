"""Serially execute the parameterized notebooks for every threshold.

Resumable: a (threshold, notebook) pair is skipped when its executed copy exists.
Executed notebooks (with outputs) go to executed/thresh_<tag>/<notebook path>;
a failure leaves <name>.FAILED.ipynb plus a line in logs/run_all.log and the run
continues with the next notebook.

usage: python run_all.py [--thresholds m2.4,m1.1] [--only substring] [--timeout SEC]
"""
import argparse
import datetime
import os
import sys
import time
import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_variants import THRESHOLDS, BASELINE, tag  # noqa: E402


def notebooks(only):
    out = []
    for top in ('classification_revised', 'regression_revised'):
        for root, _, files in os.walk(os.path.join(HERE, top)):
            for f in files:
                if f.endswith('_halo.ipynb') and not f.endswith('FAILED.ipynb') and (not only or only in f):
                    out.append(os.path.join(root, f))
    return sorted(out)


def log(msg):
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, 'logs', 'run_all.log'), 'a') as fh:
        fh.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thresholds', default=None)
    ap.add_argument('--only', default=None)
    ap.add_argument('--timeout', type=int, default=-1)
    a = ap.parse_args()
    # baseline first, then the rest in order of distance from it
    tags = a.thresholds.split(',') if a.thresholds else \
        [tag(t) for t in sorted(THRESHOLDS, key=lambda t: abs(t - BASELINE))]
    nbs = notebooks(a.only)
    log(f"start: {len(tags)} thresholds x {len(nbs)} notebooks")
    for tg in tags:
        os.environ['THRESH_TAG'] = tg
        for path in nbs:
            rel = os.path.relpath(path, HERE)
            dst = os.path.join(HERE, 'executed', f'thresh_{tg}', rel)
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            nb = nbformat.read(path, as_version=4)
            t0 = time.time()
            client = NotebookClient(nb, timeout=a.timeout, kernel_name='python3',
                                    resources={'metadata': {'path': os.path.dirname(path)}})
            try:
                client.execute()
                nbformat.write(nb, dst)
                log(f"ok     {tg} {rel} ({(time.time() - t0) / 60:.1f} min)")
            except Exception as e:  # keep going; record what failed
                nbformat.write(nb, dst.replace('.ipynb', '.FAILED.ipynb'))
                log(f"FAILED {tg} {rel}: {type(e).__name__}: {str(e)[:300]}")
    log("done")


if __name__ == '__main__':
    main()
