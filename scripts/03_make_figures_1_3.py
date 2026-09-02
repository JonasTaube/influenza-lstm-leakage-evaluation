# -*- coding: utf-8 -*-
"""
Step 3: Generate publication-quality figures
- Figure 1: TA-AFWLSTM network architecture (schematic)
- Figure 3: Weekly influenza positivity time series, 2015-2025
Output: ../figures/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np

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

# ================= Figure 1: 网络结构图 =================
def draw_figure1():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 42)
    ax.axis("off")

    # 颜色
    c_input  = "#DCE9F7"   # light blue
    c_afwm   = "#FDEBD0"   # light orange
    c_lstm   = "#E8F6EF"   # light green
    c_attn   = "#F5E6F7"   # light purple
    c_out    = "#FDEDEC"   # light red
    c_border = "#333333"

    def box(x, y, w, h, text, fc, fs=9, bold=False, sub=None):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.2",
                           linewidth=1.1, edgecolor=c_border, facecolor=fc, zorder=2)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2 + (1.2 if sub else 0), text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)
        if sub:
            ax.text(x + w/2, y + h/2 - 1.6, sub, ha="center", va="center",
                    fontsize=fs-1.5, color="#555555", zorder=3)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                            linewidth=1.3, color="#333333", zorder=1)
        ax.add_patch(a)

    # Input
    box(2, 28, 16, 11, "Input", c_input, bold=True,
        sub="Weekly positivity rates\nof 4 subtypes\n(12-week window)")
    # AFWM
    box(24, 28, 16, 11, "AFWM", c_afwm, bold=True,
        sub="Gated adaptive\nfeature weighting")
    # LSTM encoder
    box(46, 33, 17, 6, "LSTM Layer 1", c_lstm, sub="64 units")
    box(46, 25, 17, 6, "LSTM Layer 2", c_lstm, sub="64 units")
    ax.text(54.5, 31.7, "", fontsize=8)
    # Temporal attention
    box(69, 28, 16, 11, "Temporal\nAttention", c_attn, bold=True,
        sub="Softmax over\n12 time steps")
    # Output
    box(91, 28, 17, 11, "Output", c_out, bold=True,
        sub="t+1 to t+4\npositivity rates")

    # Arrows
    arrow(18, 33.5, 24, 33.5)
    arrow(40, 33.5, 46, 33.5)
    arrow(54.5, 33, 54.5, 31)          # L1 -> L2
    arrow(54.5, 25, 54.5, 23.5)        # down
    arrow(54.5, 23.5, 69, 29.5)        # L2 -> attention (diag)
    arrow(85, 33.5, 91, 33.5)

    # 底部标注：直接多输出
    ax.annotate("", xy=(99.5, 39), xytext=(99.5, 34.5),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0))
    ax.text(96.5, 40.8, "Direct multi-step output", fontsize=7.5, ha="center", color="#444444")

    # 图注辅助说明
    ax.text(2, 2, "A: LSTM cell   AFWM: Adaptive Feature Weighting Module", fontsize=7.5, color="#555555")

    plt.tight_layout()
    fig.savefig(FIG + "Figure1_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG + "Figure1_architecture.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 1 saved.")

# ================= Figure 3: 时间序列 =================
def draw_figure3():
    d = pd.read_csv("../tables/main_series_2015_2025.csv",
                    encoding="utf-8-sig", parse_dates=["week_start"])
    x = d["week_start"]
    n = len(d)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.08})

    # Panel A: weekly specimen counts
    ax = axes[0]
    ax.bar(x, d["n_samples"], width=6, color="#9DB8D2", edgecolor="none")
    ax.set_ylabel("Specimens tested\nper week")
    ax.set_ylim(0, 500)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: subtype positivity rates (stacked area) + combined rate
    ax = axes[1]
    rate_cols = [("rate_h3n2", "#C0504D", "A(H3N2)"),
                 ("rate_h1n1", "#ED7D31", "A(H1N1)pdm09"),
                 ("rate_by",   "#4F81BD", "B/Yamagata"),
                 ("rate_bv",   "#9BBB59", "B/Victoria")]
    ax.stackplot(x, *[d[c[0]].values for c in rate_cols],
                 labels=[c[2] for c in rate_cols],
                 colors=[c[1] for c in rate_cols], alpha=0.85)
    ax.plot(x, d["rate_total"], color="#1F1F1F", lw=1.1, label="Combined positivity rate", zorder=3)
    ax.set_ylabel("Weekly positivity\nrate")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)

    # x axis (years)
    years = pd.date_range("2015-01-01", "2026-01-01", freq="YS")
    ax.set_xticks(years)
    ax.set_xticklabels([f"{y.year}" for y in years])
    ax.set_xlim(d["week_start"].min(), d["week_start"].max())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=7.5, frameon=False, ncol=5,
              bbox_to_anchor=(0, 1.02))

    plt.tight_layout()
    fig.savefig(FIG + "Figure3_timeseries.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG + "Figure3_timeseries.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 saved.")

draw_figure1()
draw_figure3()
print("Done.")
