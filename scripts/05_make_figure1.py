# -*- coding: utf-8 -*-
"""
Step 5: Redraw Figure 1 - Online-Adaptive LSTM architecture + walk-forward workflow
Panel A: network structure (Input 12x4 -> 2-layer LSTM -> FC+GELU -> t+1..t+4)
Panel B: walk-forward online adaptation loop (Predict -> Observe -> Fine-tune -> next window)
Output: ../figures/Figure1_architecture.{png,tiff}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG = "../figures/"
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
})

C_INPUT = "#DCE9F7"   # light blue
C_LSTM  = "#E8F6EF"   # light green
C_FC    = "#FDEBD0"   # light orange
C_OUT   = "#FDEDEC"   # light red
C_LOOP  = "#F5E6F7"   # light purple
C_BORDER = "#333333"


def box(ax, x, y, w, h, text, fc, fs=9, bold=False, sub=None):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.2",
                       linewidth=1.1, edgecolor=C_BORDER, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2 + (1.2 if sub else 0), text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)
    if sub:
        ax.text(x + w/2, y + h/2 - 1.6, sub, ha="center", va="center",
                fontsize=fs-1.5, color="#555555", zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.3, style="-|>", connstyle=None):
    kw = dict(arrowstyle=style, mutation_scale=14, linewidth=lw, color=color, zorder=1)
    if connstyle:
        kw["connectionstyle"] = connstyle
    a = FancyArrowPatch((x1, y1), (x2, y2), **kw)
    ax.add_patch(a)


fig = plt.figure(figsize=(11, 8.2))

# ============================ Panel A: architecture ============================
axA = fig.add_axes([0.02, 0.52, 0.96, 0.46])
axA.set_xlim(0, 110)
axA.set_ylim(0, 42)
axA.axis("off")
axA.text(1, 39.5, "a", fontsize=13, fontweight="bold", ha="left", va="top")
axA.text(4, 39.5, "Online-adaptive LSTM architecture", fontsize=10.5,
         fontweight="bold", ha="left", va="top")

# Input (with mini grid icon)
box(axA, 2, 24, 18, 12, "Input", C_INPUT, bold=True,
    sub="Weekly positivity rates\nof 4 subtypes\n(12-week window)")
# mini input matrix icon
gx, gy = 14.2, 26.5
for r in range(4):
    for c in range(6):
        rect = Rectangle((gx + c*0.62, gy + r*0.62), 0.5, 0.5,
                         facecolor="#8FAFD4", edgecolor="none", zorder=3)
        axA.add_patch(rect)

# LSTM encoder
box(axA, 27, 30, 18, 6.5, "LSTM Layer 1", C_LSTM, sub="64 units")
box(axA, 27, 21.5, 18, 6.5, "LSTM Layer 2", C_LSTM, sub="64 units")
axA.text(36, 18.6, "dropout 0.1", fontsize=7.5, ha="center", color="#555555")
axA.text(47.5, 33.2, r"$h_T$", fontsize=9, style="italic", ha="left", color="#444444")

# FC + GELU
box(axA, 52, 24, 16, 12, "FC + GELU", C_FC, bold=True,
    sub="fully connected\nhidden layer")

# Output
box(axA, 75, 24, 18, 12, "Output", C_OUT, bold=True,
    sub=r"$\hat{y}_{t+1},\ \hat{y}_{t+2},\ \hat{y}_{t+3},\ \hat{y}_{t+4}$" + "\n(direct multi-step)")

# Arrows
arrow(axA, 20, 30, 27, 33.2)
arrow(axA, 36, 30, 36, 28)
arrow(axA, 45, 27.5, 52, 30)
arrow(axA, 68, 30, 75, 30)

axA.annotate("", xy=(93, 38.5), xytext=(93, 36.2),
             arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0))
axA.text(93, 39.3, "combined positivity rate,\nweeks t+1 to t+4", fontsize=7.5,
         ha="center", va="bottom", color="#444444")

# bottom note
axA.text(2, 3.5,
         "Input: 12-week sliding window of subtype-resolved positivity rates "
         "(A(H3N2), A(H1N1), B/Yamagata, B/Victoria)",
         fontsize=7.5, color="#555555")

# ============================ Panel B: walk-forward loop ============================
axB = fig.add_axes([0.02, 0.03, 0.96, 0.44])
axB.set_xlim(0, 110)
axB.set_ylim(0, 36)
axB.axis("off")
axB.text(1, 33.5, "b", fontsize=13, fontweight="bold", ha="left", va="top")
axB.text(4, 33.5, "Walk-forward online adaptation (test period)", fontsize=10.5,
         fontweight="bold", ha="left", va="top")

# three loop boxes
yB = 14
box(axB, 6, yB, 24, 10, "1. Predict", C_INPUT, bold=True,
    sub=r"forecast $\hat{y}_i$ for weeks t+1 – t+4" + "\nwith current weights")
box(axB, 43, yB, 24, 10, "2. Observe", C_FC, bold=True,
    sub="true value $y_i$ becomes\navailable the following week")
box(axB, 80, yB, 24, 10, "3. Fine-tune", C_LSTM, bold=True,
    sub="3 gradient steps on $(x_i, y_i)$\n" + r"lr $=3\times10^{-4}$")

# forward arrows
arrow(axB, 30, yB + 5, 43, yB + 5)
arrow(axB, 67, yB + 5, 80, yB + 5)

# return-loop arrow (large bottom arc)
arrow(axB, 92, yB - 0.5, 18, yB - 0.5, lw=1.6, color="#8E44AD",
      connstyle="arc3,rad=0.25")
axB.text(55, 1.2, "next test window  (i → i+1)", fontsize=8.5, ha="center",
         color="#8E44AD", fontweight="bold")

# top annotation
axB.text(55, 31, "Mirrors the rolling-forecast update of ARIMA, applied to a neural network:",
         fontsize=8.5, ha="center", color="#444444")
axB.text(55, 28.2, "the model continuously re-calibrates to the shifting distribution "
                   "(post-COVID resurgence) without retraining from scratch.",
         fontsize=8.5, ha="center", color="#444444")

# distribution-shift annotation (left-side timeline note)
axB.annotate("", xy=(38, 24), xytext=(6, 24),
             arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0))
axB.text(6, 25.5, "test period (2023–2025):", fontsize=7.5, ha="left", color="#555555")
axB.text(6, 23.2, "distribution shift", fontsize=7.5, ha="left", color="#555555")

fig.savefig(FIG + "Figure1_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG + "Figure1_architecture.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 1 (online-adaptive LSTM architecture) saved.")
