# -*- coding: utf-8 -*-
"""Revision 1: composite the regenerated base figures into the final
manuscript figure set (Figures 2-4 of the manuscript).

  Final_Figure2 = (a) leakage illusion + (b) weekly time series
  Final_Figure3 = prediction vs observed (4 horizons)
  Final_Figure4 = (a) horizon R + (b) season MAE + (c) ablation

Final_Figure1 is NOT produced here: the manuscript's Figure 1 is the
BioRender schematic supplied by the authors.
"""
import os
import sys

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

sys.stdout.reconfigure(encoding='utf-8')

FIG = '../figures/'


def load_png(name):
    return Image.open(os.path.join(FIG, name + '.png'))


def add_panel_labels(fig, labels):
    for ax, lab in zip(fig.axes, labels):
        ax.text(0.02, 0.96, f'({lab})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left',
                color='#222222', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.85))


def save(fig, name):
    fig.savefig(os.path.join(FIG, f'{name}.png'), dpi=600, facecolor='white',
                bbox_inches='tight')
    fig.savefig(os.path.join(FIG, f'{name}.pdf'), format='pdf', facecolor='white',
                bbox_inches='tight')


# ===== Final Figure 2: leakage (a) + time series (b) =====
fig2 = plt.figure(figsize=(12, 10))
ax2a = fig2.add_subplot(2, 1, 1)
ax2b = fig2.add_subplot(2, 1, 2)
ax2a.imshow(load_png('Figure2_online_adaptation'))
ax2b.imshow(load_png('Figure3_timeseries'))
for ax in (ax2a, ax2b):
    ax.axis('off')
add_panel_labels(fig2, ['a', 'b'])
fig2.tight_layout()
save(fig2, 'Final_Figure2')
plt.close(fig2)
print('Final_Figure2 done', flush=True)

# ===== Final Figure 3: prediction vs observed =====
fig3 = plt.figure(figsize=(12, 10))
ax3 = fig3.add_subplot(1, 1, 1)
ax3.imshow(load_png('Figure4_prediction_vs_observed'))
ax3.axis('off')
fig3.tight_layout()
save(fig3, 'Final_Figure3')
plt.close(fig3)
print('Final_Figure3 done', flush=True)

# ===== Final Figure 4: horizon R (a) + season (b) + ablation (c) =====
fig4 = plt.figure(figsize=(14, 12))
ax4a = fig4.add_subplot(2, 2, 1)
ax4b = fig4.add_subplot(2, 2, 2)
ax4c = fig4.add_subplot(2, 1, 2)
ax4a.imshow(load_png('Figure5_horizon_r'))
ax4b.imshow(load_png('Figure6_season'))
ax4c.imshow(load_png('Figure7_ablation'))
for ax in (ax4a, ax4b, ax4c):
    ax.axis('off')
add_panel_labels(fig4, ['a', 'b', 'c'])
fig4.tight_layout()
save(fig4, 'Final_Figure4')
plt.close(fig4)
print('Final_Figure4 done', flush=True)

print('\nFinal figures 2-4 saved to:', os.path.abspath(FIG), flush=True)
