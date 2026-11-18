# -*- coding: utf-8 -*-
"""Revision 1 figure regeneration (Scientific Reports).

Adapted from the original generate_figures.py with three fixes:
  1. Figure 2a: seed-42 curves now use seed index 0 (SEEDS[0]=42); the
     original script plotted index 2 (seed 777) while labelling it seed 42.
  2. Figure 2a: the 10-sample rolling mean is applied to ALL three curves
     (leaky / single-step / static); the original smoothed only the leaky
     curve. A pandas rolling(min_periods=1) is used so edges are not biased.
  3. Figure 2a: errors are multiplied by 100 so the y-axis truly reads
     "percentage points" (rates are stored as fractions).
  Plus: Figure 3/4 now read the modelling CSV (main_series_2015_2025.csv)
  instead of Weekly data.xlsx, and Figure 4 x-axis dates are aligned to the
  true target weeks (split + INPUT_LEN + h) instead of a naive tail slice.

Figure 1 (architecture) is NOT regenerated: the submitted manuscript uses a
BioRender figure which the authors replaced manually.
"""
import csv as _csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from genshin_style import COLORS, MODEL_PALETTE, SEASON_PALETTE, SUBTYPE_PALETTE, set_genshin_style, save_both

sys.stdout.reconfigure(encoding='utf-8')
set_genshin_style(dpi=300)

TAB = '../tables/'
FIG = '../figures/'
os.makedirs(FIG, exist_ok=True)

res = json.load(open(TAB + 'results.json', encoding='utf-8'))
np_seed = np.load(TAB + 'seed_predictions.npz', allow_pickle=True)
np_extra = np.load(TAB + 'extra_predictions.npz', allow_pickle=True)

df = pd.read_csv(TAB + 'main_series_2015_2025.csv', encoding='utf-8-sig',
                 parse_dates=['week_start'])
week_starts = list(df['week_start'])
n_samples = df['n_samples'].values.astype(float)
rate_h3n2 = df['rate_h3n2'].values.astype(float)
rate_h1n1 = df['rate_h1n1'].values.astype(float)
rate_by = df['rate_by'].values.astype(float)
rate_bv = df['rate_bv'].values.astype(float)
rate_total = df['rate_total'].values.astype(float)

INPUT_LEN, OUTPUT_LEN, TRAIN_RATIO = 12, 4, 0.70
n_win = len(df) - INPUT_LEN - OUTPUT_LEN + 1
split = int(n_win * TRAIN_RATIO)          # first test window (391)
n_test = len(df) - (split + INPUT_LEN) - OUTPUT_LEN + 1  # 168

# True target-week dates per horizon: window i, horizon h (0-based) targets
# week split + i + INPUT_LEN + h.
def horizon_dates(h):
    idx = [split + i + INPUT_LEN + h for i in range(n_test)]
    return [week_starts[j] for j in idx]


# ========== Figure 2: leakage illusion ==========
# SEEDS = [42, 123, 777, 2024, 8888] -> seed 42 is index 0.
leaky = np_extra['leaky']                     # (5, 168, 4)
static = np_seed['Static_LSTM']
single = np_seed['Online_LSTM_single_step_update']

leaky_err = np.abs(leaky[0, :, 0] - np_seed['Y_te'][:, 0]) * 100
static_err = np.abs(static[0, :, 0] - np_seed['Y_te'][:, 0]) * 100
single_err = np.abs(single[0, :, 0] - np_seed['Y_te'][:, 0]) * 100

def roll10(x):
    return pd.Series(x).rolling(10, min_periods=1).mean().values

fig2, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(roll10(leaky_err), color='#C0392B', lw=1.8,
        label='Online, leaky full-target update (10-sample rolling mean)')
ax.plot(roll10(single_err), color='#E67E22', lw=1.2,
        label='Online, leakage-free single-step update (10-sample rolling mean)')
ax.plot(roll10(static_err), color='#2E6DB4', lw=1.5,
        label='Static LSTM (10-sample rolling mean)')
ax.set_xlabel('Test sample index (chronological)', fontsize=11)
ax.set_ylabel('Absolute error, t+1 horizon (percentage points)', fontsize=11)
ax.set_title('The leakage illusion: apparent adaptation vanishes under leakage-free updates (seed 42)',
             fontsize=12, fontweight='bold', color=COLORS['abyss'])
ax.legend(loc='upper right', frameon=True, fancybox=True, facecolor='white',
          edgecolor='none', fontsize=8.5)
ax.set_xlim(-5, n_test + 5)
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_both(fig2, os.path.join(FIG, 'Figure2_online_adaptation'))
plt.close(fig2)
print('Figure2 done', flush=True)


# ========== Figure 3: time series ==========
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.4], 'hspace': 0.05})
ax3a.bar(week_starts, n_samples, color='#A8D8F0', edgecolor=COLORS['hydro'],
         width=5, lw=0.3)
ax3a.set_ylabel('Specimens tested\nper week', fontsize=10)
ax3a.set_title('Weekly influenza sentinel surveillance in Tianjin, 2015-2025',
               fontsize=12, fontweight='bold', color=COLORS['abyss'])
ax3a.set_ylim(0, 520)
ax3a.spines['top'].set_visible(False)
ax3a.spines['right'].set_visible(False)

ax3b.stackplot(week_starts, rate_h3n2, rate_h1n1, rate_by, rate_bv,
               labels=['A(H3N2)', 'A(H1N1)pdm09', 'B/Yamagata', 'B/Victoria'],
               colors=SUBTYPE_PALETTE, alpha=0.85, edgecolor='white', linewidth=0.3)
ax3b.plot(week_starts, rate_total, color=COLORS['abyss'], lw=1.2,
          label='Combined positivity rate')
ax3b.set_ylabel('Weekly positivity\nrate', fontsize=10)
ax3b.set_ylim(0, 1.0)
ax3b.legend(loc='upper left', ncol=5, frameon=True, fancybox=True, fontsize=8.5,
            bbox_to_anchor=(0.0, 1.14))
ax3b.spines['top'].set_visible(False)
ax3b.spines['right'].set_visible(False)
ax3b.xaxis.set_major_locator(mdates.YearLocator())
ax3b.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
save_both(fig3, os.path.join(FIG, 'Figure3_timeseries'))
plt.close(fig3)
print('Figure3 done', flush=True)


# ========== Figure 4: prediction vs observed ==========
obs = np_seed['Y_te']                                   # (168, 4)
pred_agg = np_seed['Agg_input_Static_LSTM'].mean(axis=0)
pred_arima = np_seed['preds_arima']
pred_static = np_seed['Static_LSTM'].mean(axis=0)

fig4, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
axes = axes.flatten()
horizon_titles = ['t+1 week horizon', 't+2 week horizon', 't+3 week horizon', 't+4 week horizon']
linestyles = ['-', '-', ':', '--']
labels = ['Observed', 'Agg-input Static LSTM', 'ARIMA (rolling)', 'Static LSTM (multi-subtype)']
series = [obs, pred_agg, pred_arima, pred_static]
colors = [COLORS['abyss'], '#C0392B', '#7F7F7F', '#2E6DB4']

for h, ax in enumerate(axes):
    hd = horizon_dates(h)
    for s, lab, ls, col in zip(series, labels, linestyles, colors):
        ax.plot(hd, s[:, h], color=col, lw=1.3 if lab == 'Observed' else 1.0,
                linestyle=ls, label=lab if h == 0 else '')
    ax.set_title(horizon_titles[h], fontsize=11, fontweight='bold', color=COLORS['abyss'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if h in [2, 3]:
        ax.set_xlabel('Date', fontsize=10)
    if h in [0, 2]:
        ax.set_ylabel('Positivity rate (%)', fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')

fig4.legend(handles=[axes[0].plot([], [], color=c, lw=1.5, ls=ls)[0]
                     for c, ls in zip(colors, linestyles)],
            labels=labels, loc='upper center', ncol=4, frameon=False,
            bbox_to_anchor=(0.5, 0.98), fontsize=9)
fig4.tight_layout(rect=[0, 0, 1, 0.96])
save_both(fig4, os.path.join(FIG, 'Figure4_prediction_vs_observed'))
plt.close(fig4)
print('Figure4 done', flush=True)


# ========== Figure 5: Pearson R by horizon ==========
models_for_r = ['Persistence', 'ARIMA', 'XGBoost', 'Static LSTM', 'Agg-input Static LSTM']
# caption-matching colours: persistence light grey, ARIMA grey, XGBoost orange,
# multi-subtype blue, aggregate-input red
R_COLORS = ['#9E9E9E', '#4D4D4D', '#E67E22', '#2E6DB4', '#C0392B']
horizons = ['t+1', 't+2', 't+3', 't+4']
fig5, ax = plt.subplots(figsize=(7, 5))
for mi, model in enumerate(models_for_r):
    r_vals = []
    for h in horizons:
        if model in res['by_horizon_5seed']:
            r_vals.append(res['by_horizon_5seed'][model][h]['R'][0])
        else:
            r_vals.append(res['overall_5seed'][model]['R'][0])
    ax.plot(horizons, r_vals, marker='o', markersize=7, lw=2.0,
            color=R_COLORS[mi], label=model)
ax.set_xlabel('Forecast horizon', fontsize=11)
ax.set_ylabel('Pearson R', fontsize=11)
ax.set_title('Correlation by forecast horizon (5-seed mean for neural models)',
             fontsize=12, fontweight='bold', color=COLORS['abyss'])
ax.set_ylim(0.5, 1.0)
ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_both(fig5, os.path.join(FIG, 'Figure5_horizon_r'))
plt.close(fig5)
print('Figure5 done', flush=True)


# ========== Figure 6: season-stratified MAE ==========
models_season = ['Persistence', 'ARIMA', 'XGBoost', 'Static LSTM', 'Agg-input Static LSTM']
fig6, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(models_season))
width = 0.32
winter = [res['season_t1_5seed'][m]['winter']['MAE'][0] for m in models_season]
summer = [res['season_t1_5seed'][m]['summer']['MAE'][0] for m in models_season]
ax.bar(x - width / 2, winter, width, label='Winter (weeks 48-10)',
       color=SEASON_PALETTE[0], edgecolor='white', linewidth=0.5)
ax.bar(x + width / 2, summer, width, label='Summer (weeks 22-34)',
       color=SEASON_PALETTE[1], edgecolor='white', linewidth=0.5)
ax.set_ylabel('MAE, t+1 horizon', fontsize=11)
ax.set_title('Winter vs summer performance (5-seed mean for neural models)',
             fontsize=12, fontweight='bold', color=COLORS['abyss'])
ax.set_xticks(x)
ax.set_xticklabels(models_season, rotation=15, ha='right')
ax.legend(loc='upper right', frameon=True, fancybox=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_both(fig6, os.path.join(FIG, 'Figure6_season'))
plt.close(fig6)
print('Figure6 done', flush=True)


# ========== Figure 7: ablation and controls ==========
with open(TAB + 'Table4_ablation.csv', encoding='utf-8') as _f:
    _rows = list(_csv.reader(_f))

abl_models, abl_mae, abl_mae_sd, abl_r, abl_r_sd = [], [], [], [], []
for _r in _rows[1:]:
    name = _r[0].strip()
    mae_str = _r[1].strip()
    r_str = _r[3].strip()
    m, s = mae_str.split('±') if '±' in mae_str else (mae_str, '0')
    rv, rs = r_str.split('±') if '±' in r_str else (r_str, '0')
    abl_models.append(name)
    abl_mae.append(float(m))
    abl_mae_sd.append(float(s))
    abl_r.append(float(rv))
    abl_r_sd.append(float(rs))

display_names = [
    'Static\nLSTM', 'Online\nsingle-step', 'Online\ndelayed', 'Online\nleaky*',
    'Periodic-\nretrain', 'Static\nMS-TFT', 'Online\nMS-TFT delayed',
    'Agg static\nLSTM', 'Agg online\ndelayed', 'Agg online\nreplay',
]
bar_colors_mae = ['#2E6DB4' if '*' not in m else COLORS['electro'] for m in abl_models]
bar_colors_r = ['#C0392B' if '*' not in m else COLORS['electro'] for m in abl_models]

fig7, ax1 = plt.subplots(figsize=(13, 5.5))
x = np.arange(len(abl_models))
width = 0.36
ax1.bar(x - width / 2, abl_mae, width, yerr=abl_mae_sd, color=bar_colors_mae,
        edgecolor='white', linewidth=0.5, capsize=2.5, label='MAE')
ax1.set_ylabel('MAE', color='#2E6DB4', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#2E6DB4')
ax1.set_ylim(0, 0.18)
ax2 = ax1.twinx()
ax2.bar(x + width / 2, abl_r, width, yerr=abl_r_sd, color=bar_colors_r,
        edgecolor='white', linewidth=0.5, capsize=2.5, label='R')
ax2.set_ylabel('R', color='#C0392B', fontsize=11)
ax2.tick_params(axis='y', labelcolor='#C0392B')
ax2.set_ylim(0, 0.95)
ax1.set_xticks(x)
ax1.set_xticklabels(display_names, fontsize=7.5)
ax1.set_title('Ablation and controls (5-seed mean ± SD); * = leaky full-target (future-information leakage)',
              fontsize=11, fontweight='bold', color=COLORS['abyss'])
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, fancybox=True)
save_both(fig7, os.path.join(FIG, 'Figure7_ablation'))
plt.close(fig7)
print('Figure7 done (10 models, incl. leaky & replay)', flush=True)

print('\nAll 6 regenerated figures saved to:', os.path.abspath(FIG), flush=True)
