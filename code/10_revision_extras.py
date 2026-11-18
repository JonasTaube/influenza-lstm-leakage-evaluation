# -*- coding: utf-8 -*-
"""
Step 10 (Revision 1, Scientific Reports): additional analyses requested by
Reviewers 1 & 2. Runs AFTER 04/08/08b and loads the corrected-boundary
predictions. Each section is independently fault-tolerant.

Sections
  A (R2-M2): online-update hyperparameter sensitivity (lr / gradient steps /
             replay buffer K)
  B (R2-M5): joint input (aggregate + 4 subtype channels), static & delayed
  C (R1-M2/R2-M1): original-split arm (391 training windows, same environment)
             to isolate the boundary-window effect from environment effects
  D (R2-M3): restructured significance analysis - per-seed DM tests, DM on the
             seed-averaged differential, block-bootstrap CIs of MAE differences,
             Holm correction across the 10 pre-specified comparisons
  E (R2-M6): performance stratified by weekly testing volume
  F (R2-M7): performance during high-activity and rapid-rise windows
  G (R2-m2): effect of clipping predictions to the valid [0, 1] range
  H (R2-m1): conventional (in-sample naive) MASE alongside the reported
             test-set-scaled MASE
  I (R2-M4): quantification of the distribution shift (train vs test,
             pre-COVID vs post-COVID; KS test, PSI, subtype composition)
  J (R2-M4): rolling-origin robustness check (three additional test folds)
"""
import json, time, warnings, copy
warnings.filterwarnings("ignore")
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import xgboost as xgb
import torch, torch.nn as nn

t0 = time.time()
TAB = "../tables/"
df = pd.read_csv(TAB + "main_series_2015_2025.csv", encoding="utf-8-sig", parse_dates=["week_start"])

INPUT_LEN, OUTPUT_LEN, TRAIN_RATIO = 12, 4, 0.70
SEEDS = [42, 123, 777, 2024, 8888]
features = ["rate_h3n2", "rate_h1n1", "rate_by", "rate_bv"]
target = "rate_total"
OUT_JSON = {}

y_tot = df[target].values
vals = df[features].values
n_win = len(df) - INPUT_LEN - OUTPUT_LEN + 1
split = int(n_win * TRAIN_RATIO)
train_end = split - OUTPUT_LEN + 1   # boundary-corrected: windows 0..387 (388)
n_samples_series = df["n_samples"].values

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def pearson(a, b):
    a = np.asarray(a).flatten(); b = np.asarray(b).flatten()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

class PlainLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden=64, out_steps=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(hidden, out_steps))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]), None

def train_base(model, Xtr, Ytr, epochs=200, lr=1e-3, batch=32, wd=1e-5):
    tr_ds = torch.utils.data.TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Ytr))
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad(); pred, _ = model(xb)
            loss = crit(pred, yb); loss.backward(); opt.step()
    return model

def online_single(model, Xte, Yte_s, steps=3, lr=3e-4, wd=1e-5):
    mdl = copy.deepcopy(model)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    preds = []
    Xte_t = torch.FloatTensor(Xte); Yte_t = torch.FloatTensor(Yte_s)
    for i in range(len(Xte)):
        mdl.eval()
        with torch.no_grad():
            p, _ = mdl(Xte_t[i:i+1]); preds.append(p.numpy()[0])
        mdl.train()
        for _ in range(steps):
            opt.zero_grad()
            p, _ = mdl(Xte_t[i:i+1])
            loss = crit(p[:, :1], Yte_t[i:i+1, :1]); loss.backward(); opt.step()
    return np.array(preds)

def online_delayed(model, Xte, Yte_s, steps=3, lr=3e-4, wd=1e-5):
    mdl = copy.deepcopy(model)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    preds = []
    Xte_t = torch.FloatTensor(Xte); Yte_t = torch.FloatTensor(Yte_s)
    lag = OUTPUT_LEN
    for i in range(len(Xte)):
        mdl.eval()
        with torch.no_grad():
            p, _ = mdl(Xte_t[i:i+1]); preds.append(p.numpy()[0])
        j = i - lag
        if j >= 0:
            mdl.train()
            for _ in range(steps):
                opt.zero_grad()
                p, _ = mdl(Xte_t[j:j+1])
                loss = crit(p, Yte_t[j:j+1]); loss.backward(); opt.step()
    return np.array(preds)

def online_replay(model, Xte, Yte_s, K=26, steps=1, lr=1e-4, wd=1e-5):
    mdl = copy.deepcopy(model)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    preds = []
    Xte_t = torch.FloatTensor(Xte); Yte_t = torch.FloatTensor(Yte_s)
    for i in range(len(Xte)):
        mdl.eval()
        with torch.no_grad():
            p, _ = mdl(Xte_t[i:i+1]); preds.append(p.numpy()[0])
        j = i - OUTPUT_LEN
        if j >= 0:
            lo = max(0, j - K + 1)
            xb = Xte_t[lo:j+1]; yb = Yte_t[lo:j+1]
            mdl.train()
            for _ in range(steps):
                opt.zero_grad()
                p, _ = mdl(xb)
                loss = crit(p, yb); loss.backward(); opt.step()
    return np.array(preds)

def build_windows(tr_end):
    X = np.array([vals[i:i+INPUT_LEN] for i in range(n_win)])
    Y = np.array([y_tot[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN] for i in range(n_win)])
    Xa = np.array([[[v] for v in y_tot[i:i+INPUT_LEN]] for i in range(n_win)])
    Xj = np.array([np.column_stack([vals[i:i+INPUT_LEN], y_tot[i:i+INPUT_LEN]]) for i in range(n_win)])
    X_tr, Y_tr = X[:tr_end], Y[:tr_end]
    X_te, Y_te = X[split:], Y[split:]
    scX = StandardScaler(); scY = StandardScaler()
    X_tr_s = scX.fit_transform(X_tr.reshape(-1, 4)).reshape(X_tr.shape)
    X_te_s = scX.transform(X_te.reshape(-1, 4)).reshape(X_te.shape)
    Y_tr_s = scY.fit_transform(Y_tr); Y_te_s = scY.transform(Y_te)
    scXa = StandardScaler()
    Xa_tr_s = scXa.fit_transform(Xa[:tr_end].reshape(-1, 1)).reshape(Xa[:tr_end].shape)
    Xa_te_s = scXa.transform(Xa[split:].reshape(-1, 1)).reshape(Xa[split:].shape)
    scXj = StandardScaler()
    Xj_tr_s = scXj.fit_transform(Xj[:tr_end].reshape(-1, 5)).reshape(Xj[:tr_end].shape)
    Xj_te_s = scXj.transform(Xj[split:].reshape(-1, 5)).reshape(Xj[split:].shape)
    return (X_tr_s, Y_tr_s, X_te_s, Y_te_s, Xa_tr_s, Xa_te_s, Xj_tr_s, Xj_te_s, Y_te)

(X_tr_s, Y_tr_s, X_te_s, Y_te_s, Xa_tr_s, Xa_te_s, Xj_tr_s, Xj_te_s, Y_te) = build_windows(train_end)

# load corrected predictions
z = np.load(TAB + "seed_predictions.npz", allow_pickle=True)
zx = np.load(TAB + "extra_predictions.npz", allow_pickle=True)
P = {k: z[k] for k in z.files}
P["Online_LSTM_leaky_update"] = zx["leaky"]
P["Agg_input_Online_LSTM_replay_update"] = zx["agg_replay"]
Y_te_npz = z["Y_te"]
mase_scale = float(z["mase_scale"])

# ===================== A: online-update sensitivity (R2-M2) =====================
try:
    print("\n===== A: online-update hyperparameter sensitivity =====", flush=True)
    sens = []
    base_rows = [
        ("Delayed multi-step", "lr 3e-4, 3 steps (main analysis)", "Online_LSTM_delayed_update"),
        ("Delayed multi-step", "lr 3e-4, 1 step", "Delayed_online_1_step"),
        ("Delayed multi-step", "lr 3e-4, 5 steps", "Delayed_online_5_steps"),
        ("Single-step", "lr 3e-4, 3 steps (main analysis)", "Online_LSTM_single_step_update"),
        ("Experience replay (agg)", "K=26, 1 step, lr 1e-4 (main analysis)", "Agg_input_Online_LSTM_replay_update"),
    ]
    for strat, cfg, key in base_rows:
        ps = P[key]
        sens.append({"Strategy": strat, "Configuration": cfg,
                     "MAE (mean±SD)": f"{np.mean([mae(Y_te_npz, p) for p in ps]):.4f}±{np.std([mae(Y_te_npz, p) for p in ps]):.4f}",
                     "R (mean±SD)": f"{np.mean([pearson(Y_te_npz, p) for p in ps]):.3f}±{np.std([pearson(Y_te_npz, p) for p in ps]):.3f}"})
    new_configs = [
        ("Delayed multi-step", "lr 1e-4, 3 steps", "delayed", dict(steps=3, lr=1e-4), "multi"),
        ("Delayed multi-step", "lr 1e-3, 3 steps", "delayed", dict(steps=3, lr=1e-3), "multi"),
        ("Single-step", "lr 1e-4, 3 steps", "single", dict(steps=3, lr=1e-4), "multi"),
        ("Single-step", "lr 1e-3, 3 steps", "single", dict(steps=3, lr=1e-3), "multi"),
        ("Single-step", "lr 3e-4, 1 step", "single", dict(steps=1, lr=3e-4), "multi"),
        ("Single-step", "lr 3e-4, 5 steps", "single", dict(steps=5, lr=3e-4), "multi"),
        ("Experience replay (agg)", "K=13, 1 step, lr 1e-4", "replay", dict(K=13, steps=1, lr=1e-4), "agg"),
        ("Experience replay (agg)", "K=52, 1 step, lr 1e-4", "replay", dict(K=52, steps=1, lr=1e-4), "agg"),
        ("Experience replay (agg)", "K=26, 1 step, lr 3e-4", "replay", dict(K=26, steps=1, lr=3e-4), "agg"),
    ]
    scY2 = StandardScaler().fit(np.array([y_tot[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN] for i in range(train_end)]))
    for strat, cfg, fn, kw, inp in new_configs:
        maes, rs = [], []
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed)
            if inp == "multi":
                m = train_base(PlainLSTM(), X_tr_s, Y_tr_s, epochs=200)
                f = {"delayed": online_delayed, "single": online_single}[fn]
                pred = f(m, X_te_s, Y_te_s, **kw)
            else:
                m = train_base(PlainLSTM(input_dim=1), Xa_tr_s, Y_tr_s, epochs=200)
                pred = online_replay(m, Xa_te_s, Y_te_s, **kw)
            pred_inv = scY2.inverse_transform(pred)
            maes.append(mae(Y_te_npz, pred_inv)); rs.append(pearson(Y_te_npz, pred_inv))
        sens.append({"Strategy": strat, "Configuration": cfg,
                     "MAE (mean±SD)": f"{np.mean(maes):.4f}±{np.std(maes):.4f}",
                     "R (mean±SD)": f"{np.mean(rs):.3f}±{np.std(rs):.3f}"})
        print(f"  {strat} | {cfg}: MAE={np.mean(maes):.4f}±{np.std(maes):.4f} R={np.mean(rs):.3f}", flush=True)
    pd.DataFrame(sens).to_csv(TAB + "Revision_S3_update_sensitivity.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["sensitivity"] = sens
except Exception as e:
    print("A failed:", e, flush=True)

# ===================== B: joint input (R2-M5) =====================
try:
    print("\n===== B: joint aggregate+subtype input =====", flush=True)
    scY2 = StandardScaler().fit(np.array([y_tot[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN] for i in range(train_end)]))
    joint = []
    for label, online in [("Joint-input Static LSTM", False), ("Joint-input Online LSTM (delayed update)", True)]:
        maes, rs = [], []
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed)
            m = train_base(PlainLSTM(input_dim=5), Xj_tr_s, Y_tr_s, epochs=200)
            if online:
                pred = online_delayed(m, Xj_te_s, Y_te_s)
            else:
                m.eval()
                with torch.no_grad():
                    pred = m(torch.FloatTensor(Xj_te_s))[0].numpy()
            pred_inv = scY2.inverse_transform(pred)
            maes.append(mae(Y_te_npz, pred_inv)); rs.append(pearson(Y_te_npz, pred_inv))
        joint.append({"Model": label,
                      "MAE (mean±SD)": f"{np.mean(maes):.4f}±{np.std(maes):.4f}",
                      "R (mean±SD)": f"{np.mean(rs):.3f}±{np.std(rs):.3f}"})
        print(f"  {label}: MAE={np.mean(maes):.4f}±{np.std(maes):.4f} R={np.mean(rs):.3f}", flush=True)
    for label, key in [("Static LSTM (multi-subtype, main analysis)", "Static_LSTM"),
                       ("Agg-input Static LSTM (main analysis)", "Agg_input_Static_LSTM")]:
        ps = P[key]
        joint.append({"Model": label,
                      "MAE (mean±SD)": f"{np.mean([mae(Y_te_npz, p) for p in ps]):.4f}±{np.std([mae(Y_te_npz, p) for p in ps]):.4f}",
                      "R (mean±SD)": f"{np.mean([pearson(Y_te_npz, p) for p in ps]):.3f}±{np.std([pearson(Y_te_npz, p) for p in ps]):.3f}"})
    pd.DataFrame(joint).to_csv(TAB + "Revision_S4_joint_input.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["joint_input"] = joint
except Exception as e:
    print("B failed:", e, flush=True)

# ===================== C: original-split arm (boundary isolation) =====================
try:
    print("\n===== C: original 391-window split arm (same environment) =====", flush=True)
    (Xo_tr_s, Yo_tr_s, Xo_te_s, Yo_te_s, Xao_tr_s, Xao_te_s, _, _, Y_te_o) = build_windows(split)
    scYo = StandardScaler().fit(np.array([y_tot[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN] for i in range(split)]))
    orig = []
    for label, dim, Xtr, Xte in [("Static LSTM (multi-subtype)", 4, Xo_tr_s, Xo_te_s),
                                  ("Agg-input Static LSTM", 1, Xao_tr_s, Xao_te_s)]:
        maes, rs = [], []
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed)
            m = train_base(PlainLSTM(input_dim=dim), Xtr, Yo_tr_s, epochs=200)
            m.eval()
            with torch.no_grad():
                pred = m(torch.FloatTensor(Xte))[0].numpy()
            pred_inv = scYo.inverse_transform(pred)
            maes.append(mae(Y_te_o, pred_inv)); rs.append(pearson(Y_te_o, pred_inv))
        orig.append({"Model": label, "Split": "original (391 train windows)",
                     "MAE (mean±SD)": f"{np.mean(maes):.4f}±{np.std(maes):.4f}",
                     "R (mean±SD)": f"{np.mean(rs):.3f}±{np.std(rs):.3f}"})
        print(f"  [original split] {label}: MAE={np.mean(maes):.4f}±{np.std(maes):.4f} R={np.mean(rs):.3f}", flush=True)
    for label, key in [("Static LSTM (multi-subtype)", "Static_LSTM"),
                       ("Agg-input Static LSTM", "Agg_input_Static_LSTM")]:
        ps = P[key]
        orig.append({"Model": label, "Split": "boundary-corrected (388 train windows)",
                     "MAE (mean±SD)": f"{np.mean([mae(Y_te_npz, p) for p in ps]):.4f}±{np.std([mae(Y_te_npz, p) for p in ps]):.4f}",
                     "R (mean±SD)": f"{np.mean([pearson(Y_te_npz, p) for p in ps]):.3f}±{np.std([pearson(Y_te_npz, p) for p in ps]):.3f}"})
    pd.DataFrame(orig).to_csv(TAB + "Revision_boundary_split_comparison.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["boundary_split_comparison"] = orig
except Exception as e:
    print("C failed:", e, flush=True)

# ===================== D: restructured significance analysis (R2-M3) =====================
try:
    print("\n===== D: DM restructure - per-seed, seed-averaged, bootstrap CI, Holm =====", flush=True)
    def dm_test(e1, e2, h=4):
        d = (np.asarray(e1)**2 - np.asarray(e2)**2).flatten()
        n = len(d); dbar = d.mean()
        g = np.var(d, ddof=1)
        for l in range(1, min(h, n-1)+1):
            g += 2*(1 - l/(h+1)) * np.mean((d[l:]-dbar)*(d[:-l]-dbar))
        if g <= 0: return float("nan"), float("nan")
        stat = dbar / np.sqrt(g/n)
        return float(stat), float(2*(1 - stats.norm.cdf(abs(stat))))

    def preds_of(name):
        mapping = {"ARIMA": "preds_arima", "XGBoost": "preds_xgb",
                   "Persistence": "preds_persist", "Seasonal naive": "preds_snaive"}
        if name in mapping: return [z[mapping[name]]]
        return list(P[name])
    def yt_for(name): return z["Y_te_a"] if name == "ARIMA" else Y_te_npz

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
    def block_bootstrap_ci(e1, e2, n_boot=2000, block=8, seed=0):
        """Circular block bootstrap of MAE(A)-MAE(B) over test windows."""
        rng = np.random.default_rng(seed)
        E1 = np.asarray(e1).reshape(-1, OUTPUT_LEN); E2 = np.asarray(e2).reshape(-1, OUTPUT_LEN)
        n = len(E1); d = np.abs(E1).mean(axis=1) - np.abs(E2).mean(axis=1)
        stats_ = []
        nb = int(np.ceil(n / block))
        for _ in range(n_boot):
            starts = rng.integers(0, n, nb)
            idx = np.concatenate([np.arange(s, s+block) % n for s in starts])[:n]
            stats_.append(d[idx].mean())
        lo, hi = np.percentile(stats_, [2.5, 97.5])
        return float(d.mean()), float(lo), float(hi)

    dm_rows = []
    pavg_list = []
    for label, a, b in COMPARISONS:
        pa, pb = preds_of(a), preds_of(b)
        ya = yt_for(a)
        # per-seed DM
        per_seed = []
        for i in range(len(pa)):
            e1 = (pa[min(i, len(pa)-1)] - ya); e2 = (pb[min(i, len(pb)-1)] - yt_for(b))
            s_, p_ = dm_test(e1, e2)
            per_seed.append((round(s_, 3), round(p_, 4)))
        # seed-averaged differential DM (single test)
        E1 = np.mean([p - ya for p in pa], axis=0)
        E2 = np.mean([p - yt_for(b) for p in pb], axis=0) if len(pb) > 1 else pb[0] - yt_for(b)
        s_avg, p_avg = dm_test(E1, E2)
        # bootstrap CI of MAE difference (seed-mean preds)
        e1m = np.mean([p - ya for p in pa], axis=0)
        e2m = (np.mean([p - yt_for(b) for p in pb], axis=0) if len(pb) > 1 else pb[0] - yt_for(b))
        diff, lo, hi = block_bootstrap_ci(e1m, e2m)
        dm_rows.append({"Comparison": label,
                        "Per-seed DM statistics": str([x[0] for x in per_seed]),
                        "Per-seed p-values": str([x[1] for x in per_seed]),
                        "Seeds significant (p<0.05)": f"{sum(1 for x in per_seed if x[1] < 0.05)}/5",
                        "Seed-averaged DM statistic": round(s_avg, 3),
                        "Seed-averaged p": round(p_avg, 4),
                        "MAE difference (A-B)": round(diff, 4),
                        "95% CI lower": round(lo, 4), "95% CI upper": round(hi, 4)})
        pavg_list.append(p_avg)
    # Holm correction across the 10 comparisons (on seed-averaged p)
    order = np.argsort(pavg_list)
    m = len(pavg_list)
    holm = {}
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pavg_list[idx])
        running = max(running, adj)
        holm[COMPARISONS[idx][0]] = round(running, 4)
    for r in dm_rows:
        r["Holm-adjusted p (seed-averaged)"] = holm[r["Comparison"]]
    pd.DataFrame(dm_rows).to_csv(TAB + "Revision_Table5_dm_restructured.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["dm_restructured"] = dm_rows
    print("  DM restructure saved (10 comparisons, per-seed + CI + Holm).", flush=True)
except Exception as e:
    print("D failed:", e, flush=True)

# ===================== E: volume stratification (R2-M6) =====================
try:
    print("\n===== E: stratification by weekly testing volume =====", flush=True)
    vol = np.array([n_samples_series[split+i+INPUT_LEN: split+i+INPUT_LEN+OUTPUT_LEN].mean()
                    for i in range(len(Y_te_npz))])
    q1, q2 = np.percentile(vol, [33.3, 66.7])
    strata = [("Low volume (<%.0f/week)" % q1, vol < q1),
              ("Medium volume", (vol >= q1) & (vol <= q2)),
              ("High volume (>%.0f/week)" % q2, vol > q2)]
    key_models = [("Persistence", "preds_persist"), ("ARIMA", "preds_arima"), ("XGBoost", "preds_xgb"),
                  ("Static LSTM", "Static_LSTM"), ("Agg-input Static LSTM", "Agg_input_Static_LSTM"),
                  ("Online LSTM (single-step)", "Online_LSTM_single_step_update")]
    rows = []
    for sname, mask in strata:
        for mname, key in key_models:
            ps = [z[key]] if key.startswith("preds_") else list(P[key])
            yt = z["Y_te_a"] if mname == "ARIMA" else Y_te_npz
            maes = [mae(yt[mask], p[mask]) for p in ps]
            rows.append({"Stratum": sname, "Windows": int(mask.sum()), "Model": mname,
                         "MAE (pooled)": f"{np.mean(maes):.4f}",
                         "MAE t+1": f"{np.mean([mae(yt[mask][:, 0], p[mask][:, 0]) for p in ps]):.4f}"})
    pd.DataFrame(rows).to_csv(TAB + "Revision_S5_volume_stratified.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["volume_strata"] = {s: {"windows": int(m.sum()), "mean_vol": float(vol[m].mean())} for s, m in strata}
    print(f"  volume tercile cut-points: {q1:.0f} / {q2:.0f} specimens per week", flush=True)
except Exception as e:
    print("E failed:", e, flush=True)

# ===================== F: high-activity / rapid-rise windows (R2-M7) =====================
try:
    print("\n===== F: high-activity and rapid-rise windows =====", flush=True)
    obs_t1 = Y_te_npz[:, 0]
    last_obs = np.array([y_tot[split + i + INPUT_LEN - 1] for i in range(len(Y_te_npz))])
    rise = obs_t1 - last_obs
    strata = [("High-activity windows (observed t+1 rate >= 20%)", obs_t1 >= 0.20),
              ("Rapid-rise windows (week-on-week increase >= 3 pp)", rise >= 0.03),
              ("Remaining windows", (obs_t1 < 0.20) & (rise < 0.03))]
    rows = []
    for sname, mask in strata:
        for mname, key in key_models:
            ps = [z[key]] if key.startswith("preds_") else list(P[key])
            yt = z["Y_te_a"] if mname == "ARIMA" else Y_te_npz
            maes = [mae(yt[mask], p[mask]) for p in ps]
            rs_ = [pearson(yt[mask][:, 0], p[mask][:, 0]) for p in ps]
            rows.append({"Stratum": sname, "Windows": int(mask.sum()), "Model": mname,
                         "MAE (pooled)": f"{np.mean(maes):.4f}",
                         "MAE t+1": f"{np.mean([mae(yt[mask][:, 0], p[mask][:, 0]) for p in ps]):.4f}",
                         "R t+1": f"{np.nanmean(rs_):.3f}"})
    pd.DataFrame(rows).to_csv(TAB + "Revision_S6_activity_stratified.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["activity_strata"] = {s: int(m.sum()) for s, m in strata}
    for s, m in strata: print(f"  {s}: {int(m.sum())} windows", flush=True)
except Exception as e:
    print("F failed:", e, flush=True)

# ===================== G: clipping check (R2-m2) =====================
try:
    print("\n===== G: clipping predictions to [0, 1] =====", flush=True)
    rows = []
    for mname, key in key_models + [("Online LSTM (delayed)", "Online_LSTM_delayed_update")]:
        ps = [z[key]] if key.startswith("preds_") else list(P[key])
        yt = z["Y_te_a"] if mname == "ARIMA" else Y_te_npz
        mae_raw = np.mean([mae(yt, p) for p in ps])
        mae_clip = np.mean([mae(yt, np.clip(p, 0.0, 1.0)) for p in ps])
        neg = np.mean([np.mean(p < 0) for p in ps])
        rows.append({"Model": mname, "MAE (unclipped)": f"{mae_raw:.4f}",
                     "MAE (clipped to [0,1])": f"{mae_clip:.4f}",
                     "Difference": f"{mae_clip - mae_raw:+.4f}",
                     "Share of predictions < 0": f"{neg*100:.2f}%"})
    pd.DataFrame(rows).to_csv(TAB + "Revision_S7_clipping.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["clipping"] = rows
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
except Exception as e:
    print("G failed:", e, flush=True)

# ===================== H: conventional MASE (R2-m1) =====================
try:
    print("\n===== H: conventional in-sample MASE =====", flush=True)
    train_weeks = (train_end - 1) + INPUT_LEN + OUTPUT_LEN - 1  # last training TARGET week (0-based, = 402)
    ytr = y_tot[:train_weeks + 1]
    # in-sample seasonal-naive error (52-week lag) over the training period
    in_sample_err = np.abs(ytr[52:] - ytr[:-52])
    conv_scale = float(in_sample_err.mean())
    rows = []
    for mname, key in key_models:
        ps = [z[key]] if key.startswith("preds_") else list(P[key])
        yt = z["Y_te_a"] if mname == "ARIMA" else Y_te_npz
        mae_m = np.mean([mae(yt, p) for p in ps])
        rows.append({"Model": mname, "Test MAE": f"{mae_m:.4f}",
                     "MASE (test-set seasonal-naive scale, reported)": f"{mae_m/mase_scale:.2f}",
                     "MASE (conventional in-sample scale)": f"{mae_m/conv_scale:.2f}"})
    pd.DataFrame(rows).to_csv(TAB + "Revision_S8_mase.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["mase"] = {"test_set_scale": round(mase_scale, 4), "in_sample_scale": round(conv_scale, 4)}
    print(f"  test-set seasonal-naive scale = {mase_scale:.4f}; in-sample scale = {conv_scale:.4f}", flush=True)
except Exception as e:
    print("H failed:", e, flush=True)

# ===================== I: distribution-shift quantification (R2-M4) =====================
try:
    print("\n===== I: distribution shift quantification =====", flush=True)
    iso_years = df["iso_year"].values; iso_weeks = df["iso_week"].values
    test_week_idx = np.arange(split + INPUT_LEN, len(df))          # weeks 403..573
    train_week_idx = np.arange(0, train_weeks + 1)
    pre_covid = (iso_years <= 2019)
    post_covid = (iso_years >= 2023) & (iso_years <= 2025)
    def dist_stats(mask, label):
        r = y_tot[mask]; v = n_samples_series[mask]
        return {"period": label, "weeks": int(mask.sum()),
                "rate_mean": round(float(r.mean()), 4), "rate_sd": round(float(r.std()), 4),
                "rate_median": round(float(np.median(r)), 4),
                "rate_p90": round(float(np.percentile(r, 90)), 4),
                "rate_max": round(float(r.max()), 4),
                "share_weeks_gt20pct": round(float((r > 0.20).mean()), 3),
                "mean_weekly_specimens": round(float(np.nanmean(v)), 1)}
    tr_mask = np.zeros(len(df), bool); tr_mask[train_week_idx] = True
    te_mask = np.zeros(len(df), bool); te_mask[test_week_idx] = True
    periods = [dist_stats(tr_mask, "Training period (2015-W01 to 2022-W37 approx.)"),
               dist_stats(te_mask, "Test period (2022-W38 to 2025-W52)"),
               dist_stats(pre_covid, "Pre-COVID (2015-2019 ISO years)"),
               dist_stats(post_covid, "Post-COVID resurgence (2023-2025 ISO years)")]
    ks_train_test = stats.ks_2samp(y_tot[tr_mask], y_tot[te_mask])
    ks_pre_post = stats.ks_2samp(y_tot[pre_covid], y_tot[post_covid])
    # PSI (10 equal-frequency bins on the training distribution)
    def psi(a, b, bins=10):
        qs = np.quantile(a, np.linspace(0, 1, bins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
        ha = np.histogram(a, qs)[0] / len(a); hb = np.histogram(b, qs)[0] / len(b)
        ha = np.clip(ha, 1e-6, None); hb = np.clip(hb, 1e-6, None)
        return float(np.sum((hb - ha) * np.log(hb / ha)))
    # subtype composition of positives (train vs test); weekly counts are
    # derived from rate_* x n_samples (the CSV stores rates, not counts)
    def comp(mask):
        n = df.loc[mask, "n_samples"].values
        counts = {c: float((df.loc[mask, "rate_" + c].values * n).sum())
                  for c in ["h3n2", "h1n1", "by", "bv"]}
        tot = sum(counts.values())
        return {k: round(v / tot, 3) for k, v in counts.items()}
    OUT_JSON["distribution_shift"] = {
        "periods": periods,
        "KS_train_vs_test": {"stat": round(float(ks_train_test.statistic), 4), "p": float(ks_train_test.pvalue)},
        "KS_precovid_vs_postcovid": {"stat": round(float(ks_pre_post.statistic), 4), "p": float(ks_pre_post.pvalue)},
        "PSI_train_vs_test": round(psi(y_tot[tr_mask], y_tot[te_mask]), 3),
        "PSI_precovid_vs_postcovid": round(psi(y_tot[pre_covid], y_tot[post_covid]), 3),
        "subtype_composition_train": comp(tr_mask),
        "subtype_composition_test": comp(te_mask),
    }
    pd.DataFrame(periods).to_csv(TAB + "Revision_S9_distribution_shift.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(OUT_JSON["distribution_shift"], indent=1, ensure_ascii=False), flush=True)
except Exception as e:
    print("I failed:", e, flush=True)

# ===================== J: rolling-origin folds (R2-M4) =====================
try:
    print("\n===== J: rolling-origin robustness (3 additional folds) =====", flush=True)
    folds = [("fold1 (test windows %d-%d)" % (n_win-312, n_win-209), n_win-312, n_win-208),
             ("fold2 (test windows %d-%d)" % (n_win-208, n_win-105), n_win-208, n_win-104),
             ("fold3 (test windows %d-%d)" % (n_win-104, n_win), n_win-104, n_win)]
    rows = []
    for fname, fs, fe in folds:
        tr_end = fs - OUTPUT_LEN                     # boundary-excluded
        d0 = df["week_start"].iloc[fs + INPUT_LEN]
        d1 = df["week_start"].iloc[fe - 1 + INPUT_LEN + OUTPUT_LEN - 1]
        print(f"  {fname}: {d0.date()} .. {d1.date()} (train windows {tr_end})", flush=True)
        Yf = np.array([y_tot[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN] for i in range(fs, fe)])
        # persistence & seasonal naive
        pf = np.array([[y_tot[fs + (i) + INPUT_LEN - 1]] * OUTPUT_LEN for i in range(fe - fs)])
        sf = np.array([[y_tot[fs + i + INPUT_LEN + s - 52] for s in range(OUTPUT_LEN)] for i in range(fe - fs)])
        # ARIMA rolling
        n_tr_ar = fs + INPUT_LEN
        ar = sm.tsa.statespace.SARIMAX(y_tot[:n_tr_ar], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52),
                                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pa = []
        for i in range(fe - fs):
            hist_end = fs + i + INPUT_LEN
            if i == 0:
                for oi in range(n_tr_ar, hist_end):
                    ar = ar.append(np.array([y_tot[oi]]), refit=False)
            else:
                ar = ar.append(np.array([y_tot[hist_end - 1]]), refit=False)
            pa.append(np.asarray(ar.get_forecast(steps=OUTPUT_LEN).predicted_mean))
        pa = np.array(pa)
        # LSTMs (static multi + agg static), 5 seeds
        Xf = np.array([vals[i:i+INPUT_LEN] for i in range(n_win)])
        Xaf = np.array([[[v] for v in y_tot[i:i+INPUT_LEN]] for i in range(n_win)])
        scX = StandardScaler(); scYf = StandardScaler()
        Xf_tr = scX.fit_transform(Xf[:tr_end].reshape(-1, 4)).reshape(Xf[:tr_end].shape)
        Xf_te = scX.transform(Xf[fs:fe].reshape(-1, 4)).reshape(Xf[fs:fe].shape)
        scXa = StandardScaler()
        Xaf_tr = scXa.fit_transform(Xaf[:tr_end].reshape(-1, 1)).reshape(Xaf[:tr_end].shape)
        Xaf_te = scXa.transform(Xaf[fs:fe].reshape(-1, 1)).reshape(Xaf[fs:fe].shape)
        Yf_tr_s = scYf.fit_transform(np.array([y_tot[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN] for i in range(tr_end)]))
        res_f = {}
        for mname, dim, Xtr_, Xte_ in [("Static LSTM", 4, Xf_tr, Xf_te), ("Agg-input Static LSTM", 1, Xaf_tr, Xaf_te)]:
            maes, rs_ = [], []
            for seed in SEEDS:
                np.random.seed(seed); torch.manual_seed(seed)
                m = train_base(PlainLSTM(input_dim=dim), Xtr_, Yf_tr_s, epochs=200)
                m.eval()
                with torch.no_grad():
                    pred = m(torch.FloatTensor(Xte_))[0].numpy()
                pred_inv = scYf.inverse_transform(pred)
                maes.append(mae(Yf, pred_inv)); rs_.append(pearson(Yf, pred_inv))
            res_f[mname] = (np.mean(maes), np.std(maes), np.mean(rs_))
        for mname, preds, det in [("Persistence", pf, True), ("Seasonal naive", sf, True), ("ARIMA (rolling)", pa, True),
                                  ("Static LSTM", None, False), ("Agg-input Static LSTM", None, False)]:
            if det:
                rows.append({"Fold": fname, "Test period": f"{d0.date()} to {d1.date()}", "Model": mname,
                             "MAE": f"{mae(Yf, preds):.4f}", "R": f"{pearson(Yf, preds):.3f}"})
            else:
                rows.append({"Fold": fname, "Test period": f"{d0.date()} to {d1.date()}", "Model": mname,
                             "MAE": f"{res_f[mname][0]:.4f}±{res_f[mname][1]:.4f}", "R": f"{res_f[mname][2]:.3f}"})
    pd.DataFrame(rows).to_csv(TAB + "Revision_S10_rolling_folds.csv", index=False, encoding="utf-8-sig")
    OUT_JSON["rolling_folds"] = rows
except Exception as e:
    print("J failed:", e, flush=True)

with open(TAB + "revision_analyses.json", "w", encoding="utf-8") as f:
    json.dump(OUT_JSON, f, ensure_ascii=False, indent=2, default=str)
print(f"\nAll revision analyses done in {time.time()-t0:.0f}s -> revision_analyses.json", flush=True)
