# -*- coding: utf-8 -*-
"""Supplementary Figure S1: boundary timeline (R1-M2 / R2-M1).

Window-construction timeline around the train-test boundary, showing for each
window its 12-week input span and 4-week target span, the first test window's
forecast origin (week f), and why the three boundary windows (389-391,
1-based) are excluded from training.
Vector PDF (fonttype 42, Times New Roman) + 300 dpi PNG.
"""
import os
import sys
from datetime import date, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Times New Roman', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 9, 'axes.linewidth': 0.8,
})

W0 = date(2014, 12, 29)          # week_start of 0-based week 0
def wk_date(w0idx):
    return W0 + timedelta(weeks=w0idx)

INPUT_LEN, OUTPUT_LEN = 12, 4
SPLIT = 391                        # 0-based index of first test window
F_ORIGIN = SPLIT + INPUT_LEN - 1   # 0-based week 402 = 12 Sep 2022

# windows to display: 1-based 384..394
rows = list(range(384, 395))       # 1-based window numbers
n = len(rows)

fig, ax = plt.subplots(figsize=(9.2, 5.6))

# shaded zones (0-based week coordinates)
ax.axvspan(383, F_ORIGIN + 0.5, color='#EAF3FB', zorder=0)
ax.axvspan(F_ORIGIN + 0.5, 420, color='#FDF1EC', zorder=0)
ax.text((383 + F_ORIGIN) / 2, n + 0.62, 'training targets admissible\n(weeks observable at or before the first test forecast origin)',
        ha='center', va='bottom', fontsize=8, color='#2E5AAC')
ax.text((F_ORIGIN + 420) / 2, n + 0.62, 'test target zone\n(evaluation targets)',
        ha='center', va='bottom', fontsize=8, color='#C23A30')

for k, w1 in enumerate(rows):      # w1 = 1-based window number
    i0 = w1 - 1                    # 0-based window index
    inp = (i0, i0 + INPUT_LEN)     # 0-based week span of inputs
    tgt = (i0 + INPUT_LEN, i0 + INPUT_LEN + OUTPUT_LEN)
    y = n - 1 - k
    if i0 < SPLIT - OUTPUT_LEN:    # retained training window
        ci, ct, lab = '#4CC2F1', '#2171A8', None
    elif i0 < SPLIT:               # excluded boundary window
        ci, ct = '#B0B7C3', '#C23A30'
    else:                          # test window
        ci, ct = '#FAB632', '#7A5C8C'
    ax.broken_barh([(inp[0], INPUT_LEN)], (y - 0.16, 0.32), facecolors=ci,
                   edgecolor='white', linewidth=0.4, zorder=2)
    ax.broken_barh([(tgt[0], OUTPUT_LEN)], (y - 0.16, 0.32), facecolors=ct,
                   edgecolor='white', linewidth=0.4, zorder=2)
    ax.text(inp[0] - 0.6, y, f'window {w1}', ha='right', va='center', fontsize=8)
    if w1 == 388:
        ax.text(tgt[1] + 0.6, y, 'last retained training window\n(target ends at origin week f)',
                ha='left', va='center', fontsize=7.5, color='#2171A8')
    if w1 == 390:
        ax.text(tgt[1] + 0.6, y, 'boundary windows 389\u2013391\nEXCLUDED from training\n(targets overlap test targets)',
                ha='left', va='center', fontsize=7.5, color='#C23A30', fontweight='bold')
    if w1 == 392:
        ax.text(tgt[1] + 0.6, y, 'first test window\n(forecast origin = week f)',
                ha='left', va='center', fontsize=7.5, color='#7A5C8C')

# forecast-origin marker
ax.axvline(F_ORIGIN + 0.5, color='#2D2D3A', lw=1.2, ls='--', zorder=3)
ax.text(F_ORIGIN + 0.5, -0.85, f"forecast origin of first test window (f)\nweek beginning {wk_date(F_ORIGIN).strftime('%d %b %Y')}",
        ha='center', va='top', fontsize=8, color='#2D2D3A')
ax.axvline(F_ORIGIN + 1.5, color='#C23A30', lw=1.0, ls=':', zorder=3)
ax.text(F_ORIGIN + 1.5, -2.15, f"first test target week\n{wk_date(F_ORIGIN + 1).strftime('%d %b %Y')}",
        ha='left', va='top', fontsize=8, color='#C23A30')

# legend
handles = [
    patches.Patch(facecolor='#4CC2F1', label='Input window (12 weeks)'),
    patches.Patch(facecolor='#2171A8', label='Target vector (4 weeks), retained'),
    patches.Patch(facecolor='#B0B7C3', label='Input window, excluded boundary'),
    patches.Patch(facecolor='#C23A30', label='Target vector, excluded boundary'),
    patches.Patch(facecolor='#FAB632', label='Test window input'),
    patches.Patch(facecolor='#7A5C8C', label='Test window target (evaluation)'),
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.16),
          ncol=3, frameon=False, fontsize=7.5)

ax.set_xlim(368, 421)
ax.set_ylim(-3.0, n + 1.7)
ax.set_yticks([])
ax.set_xticks(list(range(370, 421, 5)))
ax.set_xticklabels([f'{wk_date(w).strftime("%d %b %Y")}' for w in range(370, 421, 5)],
                   rotation=30, ha='right', fontsize=7.5)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.set_xlabel('Surveillance week (week beginning)', fontsize=9)

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(FIG, f'Supplementary_Figure_S1_boundary_timeline.{ext}'),
                dpi=300 if ext == 'png' else None, facecolor='white', bbox_inches='tight')
print('Supplementary Figure S1 saved (png+pdf)')
