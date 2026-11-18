# -*- coding: utf-8 -*-
"""
Step 1: Build weekly influenza positivity-rate series from raw sentinel
specimen records.
Input:  raw specimen CSV (column schema documented in README)
Output: ../data/tianjin_weekly_influenza.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime

RAW = "../data/raw_specimen_records.csv"
OUT = "../data/tianjin_weekly_influenza.csv"

# ---------- 1. 读取原始数据 ----------
print("读取原始 CSV ...")
df = pd.read_csv(RAW, encoding="gb18030", dtype=str, low_memory=False)
print(f"原始记录数: {len(df):,}")

# ---------- 2. Filter sentinel ILI throat-swab specimens ----------
df = df[df["source"] == "national ILI sentinel hospital"].copy()
df = df[df["specimen_type"] == "throat swab"].copy()
print(f"Sentinel ILI throat swabs: {len(df):,}")

# ---------- 3. Date field cleaning ----------
df["collect_date"] = pd.to_datetime(df["collect_date"], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=["collect_date"])
df = df[(df["collect_date"] >= "2014-01-01") & (df["collect_date"] <= "2026-03-31")]

# ISO week
iso = df["collect_date"].dt.isocalendar()
df["iso_year"] = iso["year"].astype(int)
df["iso_week"] = iso["week"].astype(int)
# week start (Monday)
df["week_start"] = df["collect_date"] - pd.to_timedelta(df["collect_date"].dt.dayofweek, unit="D")
df["week_start"] = df["week_start"].dt.normalize()

print(f"Date range: {df['collect_date'].min().date()} ~ {df['collect_date'].max().date()}")
print(f"Specimens (after filtering): {len(df):,}")

# ---------- 4. Subtype positivity flags ----------
def pos(x):
    return (x == "positive").astype(int)

df["pos_h3n2"] = pos(df["rt_a_h3"])
df["pos_h1n1"] = pos(df["rt_a_h1"])
df["pos_by"]   = pos(df["rt_b_yamagata"])
df["pos_bv"]   = pos(df["rt_b_victoria"])
# influenza-positive = any subtype positive
df["pos_any"] = ((df["rt_a_universal"] == "positive") |
                 (df["rt_b"] == "positive")).astype(int)
# COVID-19
df["pos_covid"] = (df["rt_covid19"] == "positive").astype(int)

# ---------- 5. Weekly aggregation ----------
g = df.groupby(["iso_year", "iso_week", "week_start"]).agg(
    n_samples=("specimen_id", "count"),
    n_h3n2=("pos_h3n2", "sum"),
    n_h1n1=("pos_h1n1", "sum"),
    n_by=("pos_by", "sum"),
    n_bv=("pos_bv", "sum"),
    n_any=("pos_any", "sum"),
    n_covid=("pos_covid", "sum"),
).reset_index()

# weekly positivity rate (%): subtype positives / weekly specimens
g["rate_h3n2"] = g["n_h3n2"] / g["n_samples"]
g["rate_h1n1"] = g["n_h1n1"] / g["n_samples"]
g["rate_by"]   = g["n_by"] / g["n_samples"]
g["rate_bv"]   = g["n_bv"] / g["n_samples"]
# combined positivity rate = sum of four subtype rates (matches manuscript)
g["rate_total"] = g["rate_h3n2"] + g["rate_h1n1"] + g["rate_by"] + g["rate_bv"]
g["rate_pos"]   = g["n_any"] / g["n_samples"]   # at least one positive (supplementary)

# week index (continuous from start)
g = g.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
g["week_idx"] = np.arange(1, len(g) + 1)

print(f"\nWeeks (2014-2026W13): {len(g)}")
print(f"Specimens per week: min={g['n_samples'].min()}, max={g['n_samples'].max()}, "
      f"median={g['n_samples'].median():.0f}, mean={g['n_samples'].mean():.1f}")
print(f"Combined positivity rate (sum of 4 subtypes) range: {g['rate_total'].min():.4f} ~ {g['rate_total'].max():.4f}")

# ---------- 6. Missing-week check (full week calendar) ----------
# 构建完整周历
full_weeks = pd.DataFrame()
for year in range(2014, 2027):
    for week in range(1, 54):
        # 该年该周是否真实存在
        try:
            d = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u")
        except ValueError:
            continue
        full_weeks = pd.concat([full_weeks, pd.DataFrame({"iso_year": [year], "iso_week": [week]})])
full_weeks = full_weeks.reset_index(drop=True)

# merge with data weeks
merged = full_weeks.merge(g, on=["iso_year", "iso_week"], how="left")
missing = merged[merged["n_samples"].isna()]
print(f"Full week calendar total: {len(full_weeks)}, missing weeks: {len(missing)}")
if len(missing):
    print("Missing weeks (sample):", missing.head(20)[["iso_year", "iso_week"]].values.tolist())

# ---------- 7. Output (missing weeks kept, flagged for imputation) ----------
merged = merged.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
# 生成连续周序号与周起始日期
merged["week_idx"] = np.arange(1, len(merged) + 1)
merged["week_start"] = merged.apply(
    lambda r: datetime.strptime(f"{int(r['iso_year'])}-W{int(r['iso_week']):02d}-1", "%G-W%V-%u"),
    axis=1)

out_cols = ["week_idx", "iso_year", "iso_week", "week_start",
            "n_samples", "n_h3n2", "n_h1n1", "n_by", "n_bv", "n_any", "n_covid",
            "rate_h3n2", "rate_h1n1", "rate_by", "rate_bv", "rate_total", "rate_pos"]
merged[out_cols].to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUT}")
print(f"Total rows: {len(merged)}")

# ---------- 8. Summary stats (Table 1 material) ----------
print("\n===== Yearly specimen counts =====")
print(df.groupby(df["collect_date"].dt.year)["specimen_id"].count().to_string())

# subtype positive totals
print("\n===== Subtype positive counts =====")
for name, col in [("H3N2","n_h3n2"), ("H1N1","n_h1n1"), ("B/Yamagata","n_by"),
                  ("B/Victoria","n_bv"), ("Any","n_any"), ("COVID-19","n_covid")]:
    print(f"{name}: {merged[col].sum():,}")
