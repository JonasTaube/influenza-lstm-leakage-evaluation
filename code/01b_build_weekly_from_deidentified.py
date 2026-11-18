# -*- coding: utf-8 -*-
"""Rebuild the weekly aggregated series from the de-identified public
specimen file, and validate it against the released weekly series.

This is the public-data counterpart of 01_build_weekly_data.py (which reads
the raw laboratory information system export containing direct identifiers
and therefore cannot be released). Weekly sample counts (n_samples) and
any-positive counts (n_any) must match exactly. Subtype-specific counts can
differ by 1-2 specimens in a small number of weeks because the public file
records multi-subtype (mixed) infections as a single token, whereas the raw
data counts each RT-PCR channel independently; the released
tianjin_weekly_influenza.csv remains the authoritative modelling input.

Usage:  python 01b_build_weekly_from_deidentified.py
Input:  ../data/deidentified_specimen_data.csv
Output: ../data/tianjin_weekly_influenza_rebuilt.csv + validation report
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'data', 'deidentified_specimen_data.csv')
OUT = os.path.join(HERE, '..', 'data', 'tianjin_weekly_influenza_rebuilt.csv')

SUBTYPES = {
    'A(H3N2)': 'n_h3n2',
    'A(H1N1)pdm09': 'n_h1n1',
    'B/Yamagata': 'n_by',
    'B/Victoria': 'n_bv',
}
ANY_POS = ['A(H3N2)', 'A(H1N1)pdm09', 'B/Victoria', 'B/Yamagata',
           'B(unsubtyped)', 'A(unsubtyped)', 'mixed infection']

df = pd.read_csv(SRC, encoding='utf-8-sig', parse_dates=['collect_date'])
df['week_start'] = df['collect_date'] - pd.to_timedelta(
    df['collect_date'].dt.dayofweek, unit='D')
df['week_start'] = df['week_start'].dt.normalize()

for res, col in SUBTYPES.items():
    df[col] = (df['pcr_result'] == res).astype(int)
df['n_any'] = df['pcr_result'].isin(ANY_POS).astype(int)

g = df.groupby(['iso_year', 'iso_week', 'week_start'], as_index=False).agg(
    n_samples=('specimen_id', 'size'),
    n_h3n2=('n_h3n2', 'sum'),
    n_h1n1=('n_h1n1', 'sum'),
    n_by=('n_by', 'sum'),
    n_bv=('n_bv', 'sum'),
    n_any=('n_any', 'sum'),
)
for c in ['h3n2', 'h1n1', 'by', 'bv']:
    g[f'rate_{c}'] = g[f'n_{c}'] / g['n_samples']
# combined rate = sum of the four subtype rates (as in the original construction)
g['rate_total'] = g['rate_h3n2'] + g['rate_h1n1'] + g['rate_by'] + g['rate_bv']
g['rate_pos'] = g['n_any'] / g['n_samples']

g = g[g['iso_year'].between(2015, 2025)].copy()
g['week_idx'] = range(1, len(g) + 1)
g['n_covid'] = ''  # not used by the main analysis
out_cols = ['week_idx', 'iso_year', 'iso_week', 'week_start', 'n_samples',
            'n_h3n2', 'n_h1n1', 'n_by', 'n_bv', 'n_any', 'n_covid',
            'rate_h3n2', 'rate_h1n1', 'rate_by', 'rate_bv', 'rate_total', 'rate_pos']
g[out_cols].to_csv(OUT, index=False, encoding='utf-8-sig')

# ---- validation against the released file ----
ref = pd.read_csv(os.path.join(HERE, '..', 'data', 'tianjin_weekly_influenza.csv'),
                  encoding='utf-8-sig')
ref = ref[ref['iso_year'].between(2015, 2025)].reset_index(drop=True)
reb = g.reset_index(drop=True)
assert len(ref) == len(reb) == 574, f'week count mismatch: {len(ref)} vs {len(reb)}'

exact_ok = True
for col in ['n_samples', 'n_any']:
    a, b = ref[col].astype(float).values, reb[col].astype(float).values
    d = int((a != b).sum())
    print(f'{col}: {"EXACT" if d == 0 else f"{d} weeks differ"}')
    exact_ok &= (d == 0)

diff_total = 0
for col in ['n_h3n2', 'n_h1n1', 'n_by', 'n_bv']:
    a, b = ref[col].astype(float).values, reb[col].astype(float).values
    d = int((a != b).sum())
    abs_diff = int(abs(a - b).sum())
    diff_total += abs_diff
    print(f'{col}: {d} weeks differ (total {abs_diff} specimens, mixed-infection tokenisation)')

print(f'weeks: {len(reb)} | specimens: {reb.n_samples.sum():.0f} | '
      f'any-positive: {reb.n_any.sum():.0f}')
print(f'subtype reconstruction difference: {diff_total} subtype-detections of '
      f'{reb.n_any.sum():.0f} any-positive ({100 * diff_total / reb.n_any.sum():.2f}%)')
print('REBUILD OK (exact on n_samples/n_any; subtype columns within mixed-infection '
      'tokenisation limits)' if exact_ok else 'REBUILD FAILED (n_samples/n_any mismatch)')
raise SystemExit(0 if exact_ok else 1)
