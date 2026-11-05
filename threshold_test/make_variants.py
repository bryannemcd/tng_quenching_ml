"""Build per-threshold copies of the four annuli parquets.

quenching_vector = (delta_sum_SFR < T), where delta_sum_SFR is the offset of
log Sigma_SFR from the rSFMS fitted in `Galaxy Partitioning - Annuli Revised.ipynb`
(cell 9, full-precision values; the rounded 0.7925/-7.572 flips 16 annuli at T=-2.4).
The rSFMS fit is held fixed; only T varies. Source parquets are read-only.
"""
import os
import numpy as np
import pandas as pd

SLOPE = 0.7925210196885569
INTERCEPT = -7.571978869402589
BASELINE = -2.4
# -1.1 is Bluck's quenched criterion. Cuts below ~-2.8 are degenerate: the
# resolution-floor peak of the delta distribution spans -3.3..-2.2 dex.
THRESHOLDS = [-1.1, -1.4, -1.9, -2.0, -2.1, -2.2, -2.4, -2.6, -2.8]   # -2.0/-2.1 added to bracket the KDE/GMM valley (~ -2.1)
SAMPLES = ['low_mass_central', 'high_mass_central', 'low_mass_satellite', 'high_mass_satellite']

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'data')
OUT = os.path.join(HERE, 'data')


def tag(t):
    return f"m{abs(t):.1f}"


def main():
    deltas, rows = [], []
    for s in SAMPLES:
        d = pd.read_parquet(os.path.join(SRC, f'{s}_df_annuli_revised.parquet'))
        delta = d['avg_SFR'] - (SLOPE * d['avg_s_mass'] + INTERCEPT)
        # sanity: must reproduce the stored baseline labels exactly
        assert ((delta < BASELINE).astype(d['quenching_vector'].dtype) == d['quenching_vector']).all(), s
        deltas.append(pd.DataFrame({'sample': s, 'subID': d['subID'].values,
                                    'bin_radii': d['bin_radii'].values, 'delta_sum_SFR': delta.values}))
        for t in THRESHOLDS:
            out = os.path.join(OUT, f'thresh_{tag(t)}')
            os.makedirs(out, exist_ok=True)
            v = d.copy()
            v['quenching_vector'] = (delta < t).astype(d['quenching_vector'].dtype)
            v.to_parquet(os.path.join(out, f'{s}_df_annuli_revised.parquet'), engine='pyarrow')
            rows.append(dict(threshold=t, sample=s, n=len(v), n_quenched=int(v['quenching_vector'].sum())))
    pd.concat(deltas).to_parquet(os.path.join(OUT, 'delta_sigma_sfr.parquet'), engine='pyarrow')
    summ = pd.DataFrame(rows)
    summ['frac_quenched'] = (summ['n_quenched'] / summ['n']).round(3)
    summ.to_csv(os.path.join(OUT, 'class_counts.csv'), index=False)
    print(summ.to_string(index=False))


if __name__ == '__main__':
    main()
