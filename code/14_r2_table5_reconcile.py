# -*- coding: utf-8 -*-
"""
Step 14 (Revision 2, Scientific Reports): reconcile Table 5 with Table 1.

R1 issue (Reviewers 1 & 2): the MAE-difference column of Table 5 was computed
on seed-AVERAGED PREDICTIONS (an implicit 5-seed ensemble), so it did not equal
the difference of the seed-averaged MAEs reported in Table 1
(|mean(e)| != mean(|e|)).

Fix (this script): all seed-level quantities are now averaged ACROSS SEEDS
FIRST and summarised afterwards, on the identical corrected-boundary
predictions used for Table 1:
  - per-seed DM tests                : unchanged (validation against R1 output)
  - seed-averaged DM statistic       : DM test on the per-window differential
                                       averaged across seeds (HAC lag 4)
  - MAE difference (A-B)             : mean over seeds of per-seed
                                       MAE(A) - MAE(B)  ==  Table1(A) - Table1(B)
  - bootstrap 95% CI                 : circular block bootstrap (block 8 windows,
                                       2000 resamples) of the seed-averaged
                                       per-window absolute-error difference
  - Holm adjustment                  : across the 10 pre-specified comparisons
                                       on the seed-averaged p-values

Inputs : tables/seed_predictions.npz, tables/extra_predictions.npz
Output : tables/R2_Table5_dm_reconciled.csv
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from scipy import stats

import os
TAB = "../tables/" if os.path.exists("../tables/seed_predictions.npz") else "../results/"
# (the pipeline writes its outputs to tables/; the Zenodo deposit ships the same
#  files under results/ so that this script can be verified without retraining)
OUTPUT_LEN = 4
N_BOOT, BLOCK, BOOT_SEED = 2000, 8, 0

z = np.load(TAB + "seed_predictions.npz", allow_pickle=True)
zx = np.load(TAB + "extra_predictions.npz", allow_pickle=True)

Y = z["Y_te"]  # identical to z["Y_te_a"] (verified)

P = {k: z[k] for k in z.files if z[k].ndim == 3}
P["Online_LSTM_leaky_update"] = zx["leaky"]
P["Agg_input_Online_LSTM_replay_update"] = zx["agg_replay"]

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def dm_test_diff(d, h=4):
    """Diebold-Mariano on a loss DIFFERENTIAL series d = L1 - L2,
    Newey-West HAC variance, lag 4. Identical math to the R1 dm_test."""
    d = np.asarray(d).flatten()
    n = len(d); dbar = d.mean()
    g = np.var(d, ddof=1)
    for l in range(1, min(h, n-1)+1):
        g += 2*(1 - l/(h+1)) * np.mean((d[l:]-dbar)*(d[:-l]-dbar))
    if g <= 0: return float("nan"), float("nan")
    stat = dbar / np.sqrt(g/n)
    return float(stat), float(2*(1 - stats.norm.cdf(abs(stat))))

def dm_test(e1, e2, h=4):
    """DM on squared-error differentials (per-seed tests, as in R1)."""
    return dm_test_diff(np.asarray(e1)**2 - np.asarray(e2)**2, h)

def preds_of(name):
    mapping = {"ARIMA": "preds_arima", "XGBoost": "preds_xgb",
               "Persistence": "preds_persist", "Seasonal naive": "preds_snaive"}
    if name in mapping: return [z[mapping[name]]]
    return list(P[name])

COMPARISONS = [
    ("Agg-input Static vs ARIMA", "Agg_input_Static_LSTM", "ARIMA"),
    ("Agg-input Static vs Persistence", "Agg_input_Static_LSTM", "Persistence"),
    ("Agg-input Static vs Static LSTM (multi-subtype)", "Agg_input_Static_LSTM", "Static_LSTM"),
    ("Agg-input Static vs Periodic-retrain", "Agg_input_Static_LSTM", "Periodic_retrain_LSTM"),
    ("Agg-input Static vs Agg-input Replay online", "Agg_input_Static_LSTM", "Agg_input_Online_LSTM_replay_update"),
    ("Periodic-retrain vs Static LSTM", "Periodic_retrain_LSTM", "Static_LSTM"),
    ("Single-step online vs Static LSTM", "Online_LSTM_single_step_update", "Static_LSTM"),
    ("Leaky online vs Static LSTM", "Online_LSTM_leaky_update", "Static_LSTM"),
    ("Leaky online vs Single-step online", "Online_LSTM_leaky_update", "Online_LSTM_single_step_update"),
    ("Agg-input Replay online vs ARIMA", "Agg_input_Online_LSTM_replay_update", "ARIMA"),
]

def block_bootstrap_ci(d_win, n_boot=N_BOOT, block=BLOCK, seed=BOOT_SEED):
    """Circular block bootstrap over test windows of the seed-averaged
    per-window absolute-error difference. mean(d_win) == MAE difference."""
    rng = np.random.default_rng(seed)
    n = len(d_win)
    nb = int(np.ceil(n / block))
    stats_ = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, s+block) % n for s in starts])[:n]
        stats_.append(d_win[idx].mean())
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    return float(lo), float(hi)

rows = []
pavg_list = []
for label, a, b in COMPARISONS:
    pa, pb = preds_of(a), preds_of(b)   # each: list of S predictions (168,4); S=1 if deterministic

    # ---- per-seed DM (unchanged; S tests, pairing seed i of A with seed i of B;
    #      deterministic B re-used for every seed) ----
    per_seed = []
    for i in range(len(pa)):
        e1 = pa[min(i, len(pa)-1)] - Y
        e2 = pb[min(i, len(pb)-1)] - Y
        s_, p_ = dm_test(e1, e2)
        per_seed.append((round(s_, 3), round(p_, 4)))

    # ---- seed-averaged per-window differentials ----
    # squared-error differential, averaged across seeds (flattened 672-vector)
    d_sq = np.mean([ (pa[min(i, len(pa)-1)] - Y)**2 - (pb[min(i, len(pb)-1)] - Y)**2
                     for i in range(len(pa)) ], axis=0)
    s_avg, p_avg = dm_test_diff(d_sq)  # DM of the seed-averaged differential

    # absolute-error difference, averaged across seeds (168-vector of per-window MAE diffs)
    d_abs_win = np.mean([ np.abs(pa[min(i, len(pa)-1)] - Y).mean(axis=1)
                          - np.abs(pb[min(i, len(pb)-1)] - Y).mean(axis=1)
                          for i in range(len(pa)) ], axis=0)
    mae_diff = float(d_abs_win.mean())

    # ---- cross-check: must equal difference of seed-averaged MAEs (Table 1) ----
    t1_diff = float(np.mean([mae(Y, p) for p in pa]) - np.mean([mae(Y, p) for p in pb]))
    assert abs(mae_diff - t1_diff) < 1e-12, f"{label}: {mae_diff} != {t1_diff}"

    lo, hi = block_bootstrap_ci(d_abs_win)
    rows.append({"Comparison": label,
                 "Per-seed DM statistics": str([x[0] for x in per_seed]),
                 "Per-seed p-values": str([x[1] for x in per_seed]),
                 "Seeds significant (p<0.05)": f"{sum(1 for x in per_seed if x[1] < 0.05)}/5",
                 "Seed-averaged DM statistic": round(s_avg, 3),
                 "Seed-averaged p": round(p_avg, 4),
                 "MAE difference (A-B)": round(mae_diff, 4),
                 "95% CI lower": round(lo, 4), "95% CI upper": round(hi, 4)})
    pavg_list.append(p_avg)

# ---- Holm correction on seed-averaged p ----
order = np.argsort(pavg_list); m = len(pavg_list)
holm = {}; running = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, (m - rank) * pavg_list[idx])
    running = max(running, adj)
    holm[COMPARISONS[idx][0]] = round(running, 4)
for r in rows:
    r["Holm-adjusted p (seed-averaged)"] = holm[r["Comparison"]]

df = pd.DataFrame(rows)
df.to_csv(TAB + "R2_Table5_dm_reconciled.csv", index=False, encoding="utf-8-sig")

print("=== VALIDATION: per-seed DM must reproduce R1 values exactly ===")
r1_path = os.path.join(TAB, "Revision_Table5_dm_restructured.csv")
if os.path.exists(r1_path):
    r1 = pd.read_csv(r1_path)
    ok = all(r1.loc[i, "Per-seed DM statistics"] == df.loc[i, "Per-seed DM statistics"]
             for i in range(len(r1)))
    print("per-seed DM reproduced exactly:", ok)
else:
    print("R1 reference CSV not present - skipping cross-version check "
          "(per-seed values are deterministic and were verified during revision)")
print()
print("=== R2 reconciled Table 5 ===")
print(df[["Comparison", "Seeds significant (p<0.05)", "Seed-averaged DM statistic",
          "Seed-averaged p", "MAE difference (A-B)", "95% CI lower", "95% CI upper",
          "Holm-adjusted p (seed-averaged)"]].to_string(index=False))
print()
print("smallest Holm-adjusted p:", df["Holm-adjusted p (seed-averaged)"].min())
print("any Holm < 0.05:", (df["Holm-adjusted p (seed-averaged)"] < 0.05).any())
