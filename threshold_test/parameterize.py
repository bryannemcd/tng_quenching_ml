"""Copy halo-split notebooks into threshold_test/, swapping hard-coded paths for
parameters read from the environment (THRESH_TAG). Originals are never modified.

usage: python parameterize.py [--pilot | --all]
"""
import glob
import os
import pathlib
import re
import sys
import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PILOT = 'classification_revised/rf_classification_annuli/all_centrals_rf_annuli_halo.ipynb'

PARAM_SRC = '''import os
# --- threshold_test parameters (set by run_all.py through the environment) ---
THRESH_TAG = os.environ.get('THRESH_TAG', 'm2.4')   # m2.4 = baseline cut of -2.4 dex
DATA_DIR = os.path.join('..', '..', 'data', f'thresh_{THRESH_TAG}')
OUT_ROOT = os.path.join('..', '..', 'outputs', f'thresh_{THRESH_TAG}')
'''

READ_RE = re.compile(r"pd\.read_parquet\s*\(\s*'\.\./\.\./data/([A-Za-z_]+_df_annuli_revised\.parquet)'\s*\)")
DIR_RE = re.compile(r"^(\s*)directory\s*=\s*'([a-z]+_(?:classification|regression)_outputs)'", re.M)


def sources():
    pats = ['classification_revised/*/*_halo.ipynb', 'regression_revised/*/*_halo.ipynb']
    out = []
    for p in pats:
        out += [pathlib.Path(os.path.relpath(f, ROOT)).as_posix() for f in glob.glob(os.path.join(ROOT, p))]
    return sorted(out)


def convert(rel):
    nb = nbformat.read(os.path.join(ROOT, rel), as_version=4)
    n_read = n_dir = 0
    for c in nb.cells:
        if c.cell_type != 'code':
            continue
        c.source, k = READ_RE.subn(lambda m: f"pd.read_parquet(os.path.join(DATA_DIR, '{m.group(1)}'))", c.source)
        n_read += k
        c.source, k = DIR_RE.subn(lambda m: f"{m.group(1)}directory = os.path.join(OUT_ROOT, '{m.group(2)}')", c.source)
        n_dir += k
        c.outputs, c.execution_count = [], None
    assert n_read >= 1 and n_dir == 2, (rel, n_read, n_dir)
    cell = nbformat.v4.new_code_cell(PARAM_SRC)
    cell.metadata['tags'] = ['parameters']
    nb.cells.insert(0, cell)
    dst = os.path.join(HERE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    nbformat.write(nb, dst)
    print('wrote', os.path.relpath(dst, ROOT), f'(reads={n_read}, save-dirs={n_dir})')


if __name__ == '__main__':
    todo = [PILOT] if '--pilot' in sys.argv else sources()
    for rel in todo:
        convert(rel)
