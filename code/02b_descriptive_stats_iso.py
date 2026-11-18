# -*- coding: utf-8 -*-
"""
Step 2b (Revision 1): descriptive statistics with ISO-year alignment.
Fixes the Supplementary Table S2 inconsistency flagged by Reviewer 1:
the original 02_descriptive_stats.py grouped weeks by the CALENDAR year of
the week's Monday (week_start.dt.year), which (a) shifted boundary weeks
between years (e.g. 2019-W01 counted in the 2018 row) and (b) dropped the
single "2014" group (ISO week 1 of 2015, beginning 29 Dec 2014, 195
specimens), so annual rows summed to 573 weeks / 81,226 specimens instead
of 574 / 81,421. This script regenerates S1 and S2 grouped by ISO year.
Conventions follow the published S1: overall any-subtype positivity is the
pooled rate; subtype rates are means of weekly rates.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

df = pd.read_csv("../data/tianjin_weekly_influenza.csv", encoding="utf-8-sig")
d = df[(df["iso_year"] >= 2015) & (df["iso_year"] <= 2025)].copy().reset_index(drop=True)

TAB = "../tables/"

# ============ Table S1 (conventions unchanged; verified) ============
desc = pd.DataFrame({
    "Indicator": ["Number of surveillance weeks", "Number of ILI specimens tested",
                  "Samples per week (mean, range)",
                  "Overall influenza-positive rate (any subtype, %)",
                  "Overall combined subtype positivity rate (%, mean ± SD)",
                  "A(H3N2) weekly positivity rate (%, mean ± SD)",
                  "A(H1N1) weekly positivity rate (%, mean ± SD)",
                  "B/Yamagata weekly positivity rate (%, mean ± SD)",
                  "B/Victoria weekly positivity rate (%, mean ± SD)"],
    "Value": [
        f"{len(d)}",
        f"{int(d['n_samples'].sum()):,}",
        f"{d['n_samples'].mean():.1f} ({d['n_samples'].min():.0f}–{d['n_samples'].max():.0f})",
        f"{d['n_any'].sum() / d['n_samples'].sum() * 100:.2f}",
        f"{d['rate_total'].mean()*100:.2f} ± {d['rate_total'].std()*100:.2f}",
        f"{d['rate_h3n2'].mean()*100:.2f} ± {d['rate_h3n2'].std()*100:.2f}",
        f"{d['rate_h1n1'].mean()*100:.2f} ± {d['rate_h1n1'].std()*100:.2f}",
        f"{d['rate_by'].mean()*100:.2f} ± {d['rate_by'].std()*100:.2f}",
        f"{d['rate_bv'].mean()*100:.2f} ± {d['rate_bv'].std()*100:.2f}",
    ]
})
desc.to_csv(TAB + "Table_S1_ISO.csv", index=False, encoding="utf-8-sig")
print("Table S1 (ISO):")
print(desc.to_string(index=False))

# ============ Table S2 grouped by ISO year (corrected) ============
g = d.groupby("iso_year").agg(
    weeks=("iso_week", "nunique"),
    specimens=("n_samples", "sum"),
    pos_h3n2=("n_h3n2", "sum"),
    pos_h1n1=("n_h1n1", "sum"),
    pos_by=("n_by", "sum"),
    pos_bv=("n_bv", "sum"),
    pos_any=("n_any", "sum"),
).reset_index()
g["sub4_sum"] = g[["pos_h3n2", "pos_h1n1", "pos_by", "pos_bv"]].sum(axis=1)
# B-positive but not subtyped to Yamagata/Victoria, plus any other
# influenza-positive specimens not assigned to one of the four subtypes
# (co-infections are counted once in pos_any but in each positive subtype
# column, so this residual can be slightly negative in some years).
g["pos_unassigned"] = g["pos_any"] - g["sub4_sum"]
g["pos_rate_any_%"] = (g["pos_any"] / g["specimens"] * 100).round(2)
g.columns = ["ISO year", "Surveillance weeks", "Specimens tested", "A(H3N2) positive",
             "A(H1N1) positive", "B/Yamagata positive", "B/Victoria positive",
             "Any subtype positive", "Sum of four subtypes", "B unsubtyped / other",
             "Annual positivity rate (%, any subtype)"]
g.to_csv(TAB + "Table_S2_ISO.csv", index=False, encoding="utf-8-sig")
print("\nTable S2 (ISO-year grouping):")
print(g.to_string(index=False))
print(f"\nTOTALS: weeks={g['Surveillance weeks'].sum()} specimens={g['Specimens tested'].sum()} "
      f"any-positive={g['Any subtype positive'].sum()} "
      f"({g['Any subtype positive'].sum()/g['Specimens tested'].sum()*100:.2f}%)")
assert g["Surveillance weeks"].sum() == 574
assert g["Specimens tested"].sum() == 81421
print("Assertions passed (574 weeks / 81,421 specimens).")
