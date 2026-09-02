# -*- coding: utf-8 -*-
"""
Step 2: Generate descriptive statistics tables (Table 1/2 material)
Output: ../tables/
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/tianjin_weekly_influenza.csv",
                 encoding="utf-8-sig", parse_dates=["week_start"])
d = df[(df["iso_year"] >= 2015) & (df["iso_year"] <= 2025)].copy()
d = d.reset_index(drop=True)
d["week_idx"] = np.arange(1, len(d) + 1)

TAB = "../tables/"

# ============ Table 1: 描述性统计 ============
desc = pd.DataFrame({
    "Indicator": ["Number of surveillance weeks", "Number of ILI specimens tested",
                  "Samples per week (mean, range)",
                  "Overall influenza-positive rate (any subtype, %)",
                  "Overall combined subtype positivity rate (%, mean ± SD)",
                  "H3N2 weekly positivity rate (%, mean ± SD)",
                  "H1N1 weekly positivity rate (%, mean ± SD)",
                  "B/Yamagata weekly positivity rate (%, mean ± SD)",
                  "B/Victoria weekly positivity rate (%, mean ± SD)"],
    "Value": [
        f"{len(d)}",
        f"{int(d['n_samples'].sum()):,}",
        f"{d['n_samples'].mean():.1f} ({d['n_samples'].min()}–{d['n_samples'].max()})",
        f"{d['rate_pos'].mean()*100:.2f}",
        f"{d['rate_total'].mean()*100:.2f} ± {d['rate_total'].std()*100:.2f}",
        f"{d['rate_h3n2'].mean()*100:.2f} ± {d['rate_h3n2'].std()*100:.2f}",
        f"{d['rate_h1n1'].mean()*100:.2f} ± {d['rate_h1n1'].std()*100:.2f}",
        f"{d['rate_by'].mean()*100:.2f} ± {d['rate_by'].std()*100:.2f}",
        f"{d['rate_bv'].mean()*100:.2f} ± {d['rate_bv'].std()*100:.2f}",
    ]
})
desc.to_csv(TAB + "Table1_descriptive.csv", index=False, encoding="utf-8-sig")

# ============ Yearly summary (supplementary) ============
d["year"] = d["week_start"].dt.year
yearly = d.groupby("year").agg(
    weeks=("week_idx", "count"),
    samples=("n_samples", "sum"),
    pos_h3n2=("n_h3n2", "sum"),
    pos_h1n1=("n_h1n1", "sum"),
    pos_by=("n_by", "sum"),
    pos_bv=("n_bv", "sum"),
    pos_any=("n_any", "sum"),
    pos_covid=("n_covid", "sum"),
).reset_index()
yearly["pos_rate_any_%"] = (yearly["pos_any"] / yearly["samples"] * 100).round(2)
yearly["pos_rate_total_%"] = ((yearly["pos_h3n2"] + yearly["pos_h1n1"] + yearly["pos_by"] + yearly["pos_bv"]) / yearly["samples"] * 100).round(2)
yearly.to_csv(TAB + "Table_yearly_summary.csv", index=False, encoding="utf-8-sig")
print(yearly.to_string(index=False))

# ============ Train/test split ============
n = len(d)
split = int(n * 0.7)
train = d.iloc[:split]
test = d.iloc[split:]
print(f"\n70/30 split: train {len(train)} weeks ({train['week_start'].min().date()} ~ {train['week_start'].max().date()}), "
      f"test {len(test)} weeks ({test['week_start'].min().date()} ~ {test['week_start'].max().date()})")

# save main analysis series (week index, dates)
d[["week_idx", "iso_year", "iso_week", "week_start", "n_samples",
   "rate_h3n2", "rate_h1n1", "rate_by", "rate_bv", "rate_total", "rate_pos"]].to_csv(
    TAB + "main_series_2015_2025.csv", index=False, encoding="utf-8-sig")
print("Saved main analysis series ->", TAB + "main_series_2015_2025.csv")
