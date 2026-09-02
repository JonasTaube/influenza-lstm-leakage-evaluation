# -*- coding: utf-8 -*-
"""
Step 8b: fixed post-hoc analysis (no re-training).
Loads leaky/replay predictions saved by 08_final_extras.py (extra_predictions.npz)
plus the v4 seed_predictions.npz, and computes:
- overall / by-horizon / season statistics for all 14 models
- 10 post-hoc Diebold-Mariano tests
- rewritten Tables 1/4/5
- regenerated Figures 2/4/6/7
Key fix: npz keys use 'Agg_input_*' (underscore), not 'Agginput*'.
"""
import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

t0 = time.time()
TAB = "../tables/"
FIG = "../figures/"
df = pd.read_csv(TAB + "main_series_2015_2025.csv", encoding="utf-8-sig", parse_dates=["week_start"])

INPUT_LEN, OUTPUT_LEN, TRAIN_RATIO = 12, 4, 0.70
features = ["rate_h3n2","rate_h1n1","rate_by","rate_bv"]

y_tot = df["rate_total"].values
n_win = len(df) - INPUT_LEN - OUTPUT_LEN + 1
split = int(n_win * TRAIN_RATIO)
dates = [df["week_start"].iloc[i+INPUT_LEN] for i in range(n_win)]
dates_te = dates[split:]

# ===== Load predictions =====
zx = np.load(TAB + "extra_predictions.npz")
leaky_preds = zx["leaky"]          # (5,168,4)
replay_preds = zx["agg_replay"]    # (5,168,4)

z = np.load(TAB + "seed_predictions.npz", allow_pickle=True)
Y_te = z["Y_te"]; Y_te_a = z["Y_te_a"]
preds_arima = z["preds_arima"]; preds_xgb = z["preds_xgb"]
preds_persist = z["preds_persist"]; preds_snaive = z["preds_snaive"]
mase_scale = float(z["mase_scale"])
P = {k: z[k] for k in z.files if k not in
     ["Y_te","Y_te_a","preds_arima","preds_xgb","preds_persist","preds_snaive","mase_scale"]}
P["Online_LSTM_leaky_update"] = leaky_preds
P["Agg_input_Online_LSTM_replay_update"] = replay_preds

def preds_of(name):
    if name == "ARIMA": return [preds_arima]
    if name == "XGBoost": return [preds_xgb]
    if name == "Persistence": return [preds_persist]
    if name == "Seasonal naive": return [preds_snaive]
    return list(P[name])
def yt_for(name): return Y_te_a if name == "ARIMA" else Y_te

ALL = ["Persistence","Seasonal naive","ARIMA","XGBoost",
       "Static_LSTM","Online_LSTM_single_step_update","Online_LSTM_delayed_update",
       "Online_LSTM_leaky_update","Periodic_retrain_LSTM",
       "Static_MS_TFT","Online_MS_TFT_delayed_update",
       "Agg_input_Static_LSTM","Agg_input_Online_LSTM_delayed_update",
       "Agg_input_Online_LSTM_replay_update"]
LABEL = {"Static_LSTM":"Static LSTM",
         "Online_LSTM_single_step_update":"Online LSTM (single-step update)",
         "Online_LSTM_delayed_update":"Online LSTM (delayed update)",
         "Online_LSTM_leaky_update":"Online LSTM (leaky full-target update)*",
         "Periodic_retrain_LSTM":"Periodic-retrain LSTM",
         "Static_MS_TFT":"Static MS-TFT",
         "Online_MS_TFT_delayed_update":"Online MS-TFT (delayed update)",
         "Agg_input_Static_LSTM":"Agg-input Static LSTM",
         "Agg_input_Online_LSTM_delayed_update":"Agg-input Online LSTM (delayed update)",
         "Agg_input_Online_LSTM_replay_update":"Agg-input Online LSTM (replay update)"}

def mae(a,b): return float(np.mean(np.abs(a-b)))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def pearson(a,b):
    a=np.asarray(a).flatten(); b=np.asarray(b).flatten()
    if len(a)<2 or np.std(a)==0 or np.std(b)==0: return float("nan")
    return float(np.corrcoef(a,b)[0,1])
def evaluate(yt, yp):
    return {"MAE": round(mae(yt,yp),4), "RMSE": round(rmse(yt,yp),4),
            "R": round(pearson(yt,yp),3)}
def dm_test(e1, e2, h=4):
    d = (np.asarray(e1)**2 - np.asarray(e2)**2).flatten()
    n = len(d); dbar = d.mean()
    g = np.var(d, ddof=1)
    for l in range(1, min(h, n-1)+1):
        g += 2*(1 - l/(h+1)) * np.mean((d[l:]-dbar)*(d[:-l]-dbar))
    if g <= 0: return float("nan"), float("nan")
    stat = dbar / np.sqrt(g/n)
    return float(stat), float(2*(1 - stats.norm.cdf(abs(stat))))
def dm_over_seeds(e1_list, e2_list):
    if len(e2_list) == 1:
        e2_list = e2_list * len(e1_list)
    out = [dm_test(a, b) for a, b in zip(e1_list, e2_list)]
    stats_ = [s for s, p in out if not np.isnan(p)]
    ps_ = [p for s, p in out if not np.isnan(p)]
    return {"per_seed": [[round(s,3), round(p,4)] for s, p in out],
            "mean_stat": round(float(np.mean(stats_)),3) if stats_ else None,
            "mean_p": round(float(np.mean(ps_)),4) if ps_ else None,
            "n_sig": int(sum(1 for p in ps_ if p < 0.05))}

# ===== Overall stats =====
overall = {}
for name in ALL:
    ps = preds_of(name); yt = yt_for(name)
    rs = [evaluate(yt, p) for p in ps]
    overall[name] = {k: (float(np.mean([r[k] for r in rs])),
                         float(np.std([r[k] for r in rs]))) for k in rs[0]}

# ===== By-horizon =====
by_step = {}
for name in ALL:
    ps = preds_of(name); yt = yt_for(name)
    per = [{f"t+{s+1}": evaluate(yt[:,s], p[:,s]) for s in range(OUTPUT_LEN)} for p in ps]
    by_step[name] = {h: {k: (float(np.mean([x[h][k] for x in per])),
                             float(np.std([x[h][k] for x in per]))) for k in per[0][h]}
                     for h in per[0]}

# ===== Season =====
iso_weeks = df["iso_week"].values[INPUT_LEN + split : INPUT_LEN + split + len(Y_te)]
m_w = ((iso_weeks >= 48) | (iso_weeks <= 10)).astype(bool)
m_s = ((iso_weeks >= 22) & (iso_weeks <= 34)).astype(bool)
season_res = {}
for name in ALL:
    ps = preds_of(name); yt = yt_for(name)
    per = [{se: evaluate(yt[mask][:,0], p[mask][:,0])
            for se, mask in [("winter", m_w), ("summer", m_s)]} for p in ps]
    season_res[name] = {se: {k: (float(np.mean([x[se][k] for x in per])),
                                 float(np.std([x[se][k] for x in per])))
                             for k in per[0][se]} for se in per[0]}

# ===== DM tests =====
def errors_of(name):
    ps = preds_of(name); yt = yt_for(name)
    return [(p - yt).flatten() for p in ps]
dm = {}
dm["Agg-input Static vs ARIMA"] = dm_over_seeds(errors_of("Agg_input_Static_LSTM"), errors_of("ARIMA"))
dm["Agg-input Static vs Persistence"] = dm_over_seeds(errors_of("Agg_input_Static_LSTM"), errors_of("Persistence"))
dm["Agg-input Static vs Static LSTM (multi-subtype)"] = dm_over_seeds(errors_of("Agg_input_Static_LSTM"), errors_of("Static_LSTM"))
dm["Agg-input Static vs Periodic-retrain"] = dm_over_seeds(errors_of("Agg_input_Static_LSTM"), errors_of("Periodic_retrain_LSTM"))
dm["Agg-input Static vs Agg-input Replay online"] = dm_over_seeds(errors_of("Agg_input_Static_LSTM"), errors_of("Agg_input_Online_LSTM_replay_update"))
dm["Periodic-retrain vs Static LSTM"] = dm_over_seeds(errors_of("Periodic_retrain_LSTM"), errors_of("Static_LSTM"))
dm["Single-step online vs Static LSTM"] = dm_over_seeds(errors_of("Online_LSTM_single_step_update"), errors_of("Static_LSTM"))
dm["Leaky online vs Static LSTM"] = dm_over_seeds(errors_of("Online_LSTM_leaky_update"), errors_of("Static_LSTM"))
dm["Leaky online vs Single-step online"] = dm_over_seeds(errors_of("Online_LSTM_leaky_update"), errors_of("Online_LSTM_single_step_update"))
dm["Agg-input Replay online vs ARIMA"] = dm_over_seeds(errors_of("Agg_input_Online_LSTM_replay_update"), errors_of("ARIMA"))
print("DM tests:", json.dumps(dm, indent=1), flush=True)

# ===== Tables =====
def fmt(pair, nd=4):
    return f"{pair[0]:.{nd}f}±{pair[1]:.{nd}f}" if pair[1] > 0 else f"{pair[0]:.{nd}f}"

t1_rows = []
for name in ALL:
    o = overall[name]
    disp = LABEL.get(name, name)
    t1_rows.append({"Model": disp,
        "MAE (mean±SD)": fmt(o["MAE"]),
        "RMSE (mean±SD)": fmt(o["RMSE"]),
        "MASE": f"{o['MAE'][0]/mase_scale:.2f}",
        "R (mean±SD)": fmt(o["R"], 3)})
pd.DataFrame(t1_rows).to_csv(TAB+"Table1_model_comparison.csv", index=False)

t2_rows = []
for name in ALL:
    ps = preds_of(name); yt = yt_for(name)
    n_seeds = len(ps)
    for s in range(OUTPUT_LEN):
        rs = [evaluate(yt[:,s], p[:,s]) for p in ps]
        t2_rows.append({"Horizon": f"t+{s+1}", "Model": LABEL.get(name, name),
            "MAE": fmt((np.mean([r['MAE'] for r in rs]), np.std([r['MAE'] for r in rs]))),
            "RMSE": fmt((np.mean([r['RMSE'] for r in rs]), np.std([r['RMSE'] for r in rs]))),
            "R": fmt((np.mean([r['R'] for r in rs]), np.std([r['R'] for r in rs])), 3)})
pd.DataFrame(t2_rows).to_csv(TAB+"Table2_by_horizon.csv", index=False)

t3_rows = []
for name in ALL:
    ps = preds_of(name); yt = yt_for(name)
    for se, mask in [("winter", m_w), ("summer", m_s)]:
        rs = [evaluate(yt[mask][:,0], p[mask][:,0]) for p in ps]
        t3_rows.append({"Season": se.capitalize(), "Model": LABEL.get(name, name),
            "MAE": fmt((np.mean([r['MAE'] for r in rs]), np.std([r['MAE'] for r in rs]))),
            "RMSE": fmt((np.mean([r['RMSE'] for r in rs]), np.std([r['RMSE'] for r in rs]))),
            "R": fmt((np.mean([r['R'] for r in rs]), np.std([r['R'] for r in rs])), 3)})
pd.DataFrame(t3_rows).to_csv(TAB+"Table3_season.csv", index=False)

t4_rows = []
for m in ["Static_LSTM","Online_LSTM_single_step_update","Online_LSTM_delayed_update",
          "Online_LSTM_leaky_update","Periodic_retrain_LSTM","Static_MS_TFT",
          "Online_MS_TFT_delayed_update","Agg_input_Static_LSTM",
          "Agg_input_Online_LSTM_delayed_update","Agg_input_Online_LSTM_replay_update"]:
    o = overall[m]
    t4_rows.append({"Model": LABEL[m],
        "MAE (mean±SD)": fmt(o["MAE"]), "RMSE (mean±SD)": fmt(o["RMSE"]),
        "R (mean±SD)": fmt(o["R"], 3)})
pd.DataFrame(t4_rows).to_csv(TAB+"Table4_ablation.csv", index=False)

t5_rows = []
for k, v in dm.items():
    t5_rows.append({"Comparison": k, "Mean DM statistic": v["mean_stat"],
                    "Mean p-value": v["mean_p"],
                    "Seeds significant (p<0.05)": f"{v['n_sig']}/5"})
pd.DataFrame(t5_rows).to_csv(TAB+"Table5_dm_tests.csv", index=False)
print("Saved Tables 1/2/3/4/5.", flush=True)

# ===== Figures =====
plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.linewidth":0.8,"axes.edgecolor":"#222222",
    "xtick.color":"#222222","ytick.color":"#222222",
    "axes.labelcolor":"#222222","text.color":"#222222"})

# Figure 2: leakage demonstration (leaky vs single-step vs static, seed 42)
p_static42 = P["Static_LSTM"][0]
p_single42 = P["Online_LSTM_single_step_update"][0]
leaky42 = leaky_preds[0]
fig, ax = plt.subplots(figsize=(8, 4.5))
prog_leak = np.abs(leaky42[:,0] - Y_te[:,0])
prog_single = np.abs(p_single42[:,0] - Y_te[:,0])
prog_stat = np.abs(p_static42[:,0] - Y_te[:,0])
ax.plot(pd.Series(prog_leak).rolling(10, min_periods=1).mean(), color="#C0504D", lw=1.8,
        label="Online, leaky full-target update (10-sample rolling mean)")
ax.plot(pd.Series(prog_single).rolling(10, min_periods=1).mean(), color="#ED7D31", lw=1.8,
        label="Online, leakage-free single-step update")
ax.plot(pd.Series(prog_stat).rolling(10, min_periods=1).mean(), color="#4F81BD", lw=1.8,
        label="Static LSTM")
ax.set_xlabel("Test sample index (chronological)")
ax.set_ylabel("Absolute error, t+1 horizon (percentage points)")
ax.set_title("The leakage illusion: apparent adaptation vanishes under leakage-free updates (seed 42)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG+"Figure2_online_adaptation.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure2_online_adaptation.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 2 saved.", flush=True)

# Figure 4: observed + Agg-static + ARIMA + Static(multi)
p_agg42 = P["Agg_input_Static_LSTM"][0]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
for s, ax in enumerate(axes.flat):
    ax.plot(dates_te, Y_te[:, s], color="#333333", lw=1.2, label="Observed")
    ax.plot(dates_te, p_agg42[:, s], color="#C0504D", lw=1.0, label="Agg-input Static LSTM")
    ax.plot(dates_te, preds_arima[:, s], color="#888888", lw=0.9,
            label="ARIMA (rolling)", alpha=0.8, linestyle=":")
    ax.plot(dates_te, p_static42[:, s], color="#4F81BD", lw=0.8,
            label="Static LSTM (multi-subtype)", alpha=0.7, linestyle="--")
    ax.set_title(f"t+{s+1} week horizon")
    ax.set_ylabel("Positivity rate (%)")
    ax.spines[["top","right"]].set_visible(False)
    if s == 0: ax.legend(frameon=False, fontsize=7)
fig.autofmt_xdate()
fig.savefig(FIG+"Figure4_prediction_vs_observed.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure4_prediction_vs_observed.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 4 saved.", flush=True)

# Figure 6: horizon-stratified R incl. persistence + agg-static
fig, ax = plt.subplots(figsize=(8, 4.5))
horizons = [f"t+{s+1}" for s in range(OUTPUT_LEN)]
models_to_plot = [("Persistence","#BBBBBB"),("ARIMA","#888888"),("XGBoost","#ED7D31"),
                  ("Static_LSTM","#4F81BD"),("Agg_input_Static_LSTM","#C0504D")]
for m, c in models_to_plot:
    rs = [by_step[m][h]["R"][0] for h in horizons]
    ax.plot(horizons, rs, marker="o", color=c, lw=1.5, label=LABEL.get(m, m))
ax.set_ylabel("Pearson R"); ax.set_xlabel("Forecast horizon")
ax.set_title("Correlation by forecast horizon (5-seed mean for neural models)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(FIG+"Figure6_horizon_r.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure6_horizon_r.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 6 saved.", flush=True)

# Figure 7: winter vs summer incl. persistence + agg-static
fig, ax = plt.subplots(figsize=(8, 4.5))
models_s = ["Persistence","ARIMA","XGBoost","Static_LSTM","Agg_input_Static_LSTM"]
x = np.arange(len(models_s)); w = 0.35
winter_mae = [season_res[m]["winter"]["MAE"][0] for m in models_s]
summer_mae = [season_res[m]["summer"]["MAE"][0] for m in models_s]
ax.bar(x - w/2, winter_mae, w, color="#4F81BD", label="Winter (weeks 48-10)")
ax.bar(x + w/2, summer_mae, w, color="#ED7D31", label="Summer (weeks 22-34)")
ax.set_xticks(x); ax.set_xticklabels([LABEL.get(m, m) for m in models_s], fontsize=7.5, rotation=15)
ax.set_ylabel("MAE, t+1 horizon"); ax.set_title("Winter vs summer performance (5-seed mean for neural models)")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(FIG+"Figure7_season.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure7_season.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 7 saved.", flush=True)

# ===== summary printout =====
print("\n===== FINAL OVERALL (5-seed mean±SD) =====", flush=True)
for name in ALL:
    o = overall[name]
    print(f"{LABEL.get(name, name):45s} MAE={o['MAE'][0]:.4f}±{o['MAE'][1]:.4f} "
          f"RMSE={o['RMSE'][0]:.4f} R={o['R'][0]:.3f}±{o['R'][1]:.3f} "
          f"MASE={o['MAE'][0]/mase_scale:.2f}", flush=True)
print(f"\nDone in {time.time()-t0:.0f}s")
