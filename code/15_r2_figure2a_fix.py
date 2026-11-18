# -*- coding: utf-8 -*-
"""
Step 15 (Revision 2, Scientific Reports): fix Figure 2 layout.

R1 issue (Reviewer 2): "Figure 2a has text and legend overlap with the graph."
The in-axes legend covered the blue/orange/red curves near their peaks.

Fixes:
  - Panel (a): legend moved BELOW the axes (outside the plotting area, ncol=3,
    shortened labels); the y-upper region is left free for the curves.
  - Panel (b): the five-item series legend (previously anchored into the top
    subplot, overlapping the specimen bars) is likewise moved below the panel.
  - Panel labels (a)/(b) are placed OUTSIDE the embedded images.
  Panel (a) is then re-composited with the time-series panel (b) into
  Final_Figure2, otherwise identically to 12_combine_figures_rev.py.

Inputs : ../tables/main_series_2015_2025.csv,
         ../tables/seed_predictions.npz, ../tables/extra_predictions.npz
Output : ../figures/Figure2_online_adaptation.{png,pdf} (fixed panel a)
         ../figures/Figure3_timeseries.{png,pdf} (fixed panel b)
         ../figures/Final_Figure2.{png,pdf} (composite a+b)
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

from genshin_style import COLORS, SUBTYPE_PALETTE, set_genshin_style, save_both

sys.stdout.reconfigure(encoding='utf-8')
set_genshin_style(dpi=300)

TAB = '../tables/'
FIG = '../figures/'
os.makedirs(FIG, exist_ok=True)

np_seed = np.load(TAB + 'seed_predictions.npz', allow_pickle=True)
np_extra = np.load(TAB + 'extra_predictions.npz', allow_pickle=True)

n_test = np_seed['Y_te'].shape[0]          # 168 test windows

# SEEDS = [42, 123, 777, 2024, 8888] -> seed 42 is index 0
leaky = np_extra['leaky']                  # (5, 168, 4)
static = np_seed['Static_LSTM']
single = np_seed['Online_LSTM_single_step_update']

leaky_err = np.abs(leaky[0, :, 0] - np_seed['Y_te'][:, 0]) * 100
static_err = np.abs(static[0, :, 0] - np_seed['Y_te'][:, 0]) * 100
single_err = np.abs(single[0, :, 0] - np_seed['Y_te'][:, 0]) * 100

def roll10(x):
    return pd.Series(x).rolling(10, min_periods=1).mean().values

fig2, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(roll10(leaky_err), color='#C0392B', lw=1.8,
        label='Leaky full-target update')
ax.plot(roll10(single_err), color='#E67E22', lw=1.2,
        label='Leakage-free single-step update')
ax.plot(roll10(static_err), color='#2E6DB4', lw=1.5,
        label='Static LSTM')
ax.set_xlabel('Test sample index (chronological)', fontsize=11)
ax.set_ylabel('Absolute error, t+1 horizon\n(percentage points)', fontsize=11)
ax.set_title('The leakage illusion: apparent adaptation vanishes under leakage-free updates (seed 42)',
             fontsize=12, fontweight='bold', color=COLORS['abyss'], pad=10)
# legend OUTSIDE the axes, below the x-axis label region - no curve overlap
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3,
          frameon=False, fontsize=9, columnspacing=1.4, handlelength=1.8)
ax.set_xlim(-5, n_test + 5)
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig2.subplots_adjust(bottom=0.24)
save_both(fig2, os.path.join(FIG, 'Figure2_online_adaptation'))
plt.close(fig2)
print('Figure2 (panel a, fixed legend) done', flush=True)

# ===== panel (b): time series with legend moved below the panel =====
df = pd.read_csv(TAB + 'main_series_2015_2025.csv', encoding='utf-8-sig',
                 parse_dates=['week_start'])
week_starts = list(df['week_start'])
n_samples = df['n_samples'].values.astype(float)
rate_h3n2 = df['rate_h3n2'].values.astype(float)
rate_h1n1 = df['rate_h1n1'].values.astype(float)
rate_by = df['rate_by'].values.astype(float)
rate_bv = df['rate_bv'].values.astype(float)
rate_total = df['rate_total'].values.astype(float)

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.4], 'hspace': 0.18})
ax3a.bar(week_starts, n_samples, color='#A8D8F0', edgecolor=COLORS['hydro'],
         width=5, lw=0.3)
ax3a.set_ylabel('Specimens tested\nper week', fontsize=10)
ax3a.set_title('Weekly influenza sentinel surveillance in Tianjin, 2015-2025',
               fontsize=12, fontweight='bold', color=COLORS['abyss'], pad=10)
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
# legend OUTSIDE, below the bottom panel - no overlap with the specimen bars
ax3b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=5,
            frameon=False, fontsize=8.5, columnspacing=1.2, handlelength=1.5)
ax3b.spines['top'].set_visible(False)
ax3b.spines['right'].set_visible(False)
ax3b.xaxis.set_major_locator(mdates.YearLocator())
ax3b.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig3.subplots_adjust(bottom=0.16)
save_both(fig3, os.path.join(FIG, 'Figure3_timeseries'))
plt.close(fig3)
print('Figure3 (panel b, fixed legend) done', flush=True)

# ===== re-composite Final Figure 2: (a) leakage illusion + (b) time series =====
figc = plt.figure(figsize=(12, 10))
ax_a = figc.add_subplot(2, 1, 1)
ax_b = figc.add_subplot(2, 1, 2)
ax_a.imshow(Image.open(os.path.join(FIG, 'Figure2_online_adaptation.png')))
ax_b.imshow(Image.open(os.path.join(FIG, 'Figure3_timeseries.png')))
for ax in (ax_a, ax_b):
    ax.axis('off')
for ax, lab in zip(figc.axes, ['a', 'b']):
    ax.text(-0.03, 1.02, f'({lab})', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right',
            color='#222222', zorder=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.85))
figc.tight_layout()
figc.savefig(os.path.join(FIG, 'Final_Figure2.png'), dpi=600,
             facecolor='white', bbox_inches='tight')
figc.savefig(os.path.join(FIG, 'Final_Figure2.pdf'), format='pdf',
             facecolor='white', bbox_inches='tight')
plt.close(figc)
print('Final_Figure2 composite done', flush=True)
