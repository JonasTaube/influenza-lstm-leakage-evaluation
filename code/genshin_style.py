# -*- coding: utf-8 -*-
"""原神主题 matplotlib 样式配置。
配色灵感来自七元素（风/岩/雷/草/水/火/冰）及部分代表性角色。
提供统一 rcParams + 调色板，供 IJID 论文配图使用。
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ===== 原神元素配色（RGB hex）=====
# 颜色名 -> 元素/来源
COLORS = {
    # 元素主色
    'anemo':       '#74C2A8',  # 风 - 温迪/琴
    'geo':         '#FAB632',  # 岩 - 钟离
    'electro':     '#AF8EC1',  # 雷 - 雷电将军
    'dendro':      '#A7C23F',  # 草 - 纳西妲
    'hydro':       '#4CC2F1',  # 水 - 芙宁娜/公子
    'pyro':        '#EF7A35',  # 火 - 胡桃/可莉
    'cryo':        '#9FD6E3',  # 冰 - 神里绫华/优菈

    # 辅助色
    'abyss':       '#2D2D3A',  # 深紫灰，用于主线条
    'mondstadt':   '#5B7C99',  # 蒙德蓝灰
    'liyue':       '#8B6914',  # 璃月金棕
    'inazuma':     '#7A5C8C',  # 稻妻紫
    'sumeru':      '#6B8E23',  # 须弥绿
    'fontaine':    '#2E5AAC',  # 枫丹蓝
    'natlan':      '#C23A30',  # 纳塔红
    'snezhnaya':   '#A0B8C8',  # 至冬冰蓝

    # 明度变体（用于 fill/edge）
    'anemo_light': '#B8E0D0',
    'geo_light':   '#FDE4A6',
    'electro_light':'#D4C2E0',
    'dendro_light':'#D4E5A0',
    'hydro_light': '#A9E0F9',
    'pyro_light':  '#F9C4A0',
    'cryo_light':  '#D0ECF3',
}

# 论文常用模型数量较少的场景：按顺序取色
MODEL_PALETTE = [
    COLORS['hydro'],   # 水 - 主模型/观测
    COLORS['pyro'],    # 火
    COLORS['anemo'],   # 风
    COLORS['electro'], # 雷
    COLORS['geo'],     # 岩
    COLORS['dendro'],  # 草
    COLORS['cryo'],    # 冰
]

#  subtype 颜色：A(H3N2)/A(H1N1)/B/Yamagata/B/Victoria
SUBTYPE_PALETTE = [
    COLORS['pyro'],    # H3N2 火红
    COLORS['geo'],     # H1N1 岩金
    COLORS['hydro'],   # B/Yamagata 水蓝
    COLORS['dendro'],  # B/Victoria 草绿
]

# 季节/正负对比
SEASON_PALETTE = [COLORS['cryo'], COLORS['geo']]  # 冬冰蓝 vs 夏岩金

# 观测 vs 预测：黑/深灰 vs 水蓝
OBS_PRED = ['#1A1A1A', COLORS['hydro'], COLORS['pyro'], COLORS['anemo'], COLORS['electro']]


def set_genshin_style(font='DejaVu Sans', dpi=300, figsize=(8, 5.5)):
    """设置统一 matplotlib 样式。"""
    plt.rcParams.update({
        # 字体
        'font.family': 'sans-serif',
        'font.sans-serif': [font, 'Arial', 'Helvetica'],
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,

        # 画布
        'figure.dpi': dpi,
        'figure.figsize': figsize,
        'savefig.dpi': dpi,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # 轴线
        'axes.edgecolor': '#444444',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.color': '#444444',
        'ytick.color': '#444444',

        # 网格
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,

        # 图例
        'legend.frameon': False,
        'legend.loc': 'best',

        # 文字
        'text.color': '#222222',
        'axes.labelcolor': '#222222',
        'xtick.labelcolor': '#444444',
        'ytick.labelcolor': '#444444',
    })


def save_both(fig, path_stem):
    """同时保存 PNG（600 dpi 高清）和 PDF（矢量）。"""
    fig.savefig(f'{path_stem}.png', dpi=600, facecolor='white', bbox_inches='tight')
    fig.savefig(f'{path_stem}.pdf', format='pdf', facecolor='white', bbox_inches='tight')
