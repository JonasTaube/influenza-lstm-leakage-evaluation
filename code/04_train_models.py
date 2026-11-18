# -*- coding: utf-8 -*-
"""
Step 4 (v4): reviewer-revised pipeline, round 2
M1: two leakage-free online variants:
    (a) single-step update  - fine-tune on the newly observed first-step value only
    (b) delayed multi-step update - fine-tune on window i-OUTPUT_LEN's FULL 4-step
        target (fully observed before predicting window i; no future info)
M2: persistence + seasonal-naive baselines; MASE
M3: periodic full-retraining control (every 13 windows, 4-week observation lag)
M4: aggregate-input ablation (single-channel total rate)
M5: all tables/figures = 5-seed mean; Diebold-Mariano tests
M6: gradient-step sensitivity (1/3/5) for the delayed update
M7: Figure 2 dual curves (online vs static per-sample error)
Also: per-seed predictions saved to npz; fixed JSON serialization for
deterministic models.
"""
import json, time, warnings, copy
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch, torch.nn as nn

t0 = time.time()
TAB = "../tables/"
FIG = "../figures/"
df = pd.read_csv(TAB + "main_series_2015_2025.csv", encoding="utf-8-sig", parse_dates=["week_start"])
print("Data shape:", df.shape, flush=True)

INPUT_LEN, OUTPUT_LEN, TRAIN_RATIO = 12, 4, 0.70
SEEDS = [42, 123, 777, 2024, 8888]
features = ["rate_h3n2","rate_h1n1","rate_by","rate_bv"]
target = "rate_total"

y_tot = df[target].values
vals = df[features].values
n_win = len(df) - INPUT_LEN - OUTPUT_LEN + 1
split = int(n_win * TRAIN_RATIO)
# --- Revision 1 boundary-leakage fix (Scientific Reports) ---
# The first test window (index `split`) has its forecast origin at week
# split+INPUT_LEN-1 (0-based). Training window i has targets ending at week
# i+INPUT_LEN+OUTPUT_LEN-1; to guarantee that no training target extends
# beyond the first test forecast origin we require
#   i+INPUT_LEN+OUTPUT_LEN-1 <= split+INPUT_LEN-1  =>  i <= split-OUTPUT_LEN.
# The three boundary windows (indices split-3 .. split-1) are therefore
# excluded from training (391 -> 388 windows); the 168 test windows and the
# seasonal-naive/ARIMA baselines (which only use observations available at
# each forecast origin) are unchanged.
train_end = split - OUTPUT_LEN + 1  # exclusive slice bound: keeps windows 0..387 (388 windows)

X = np.array([vals[i:i+INPUT_LEN] for i in range(n_win)])
Y = np.array([y_tot[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN] for i in range(n_win)])
dates = [df["week_start"].iloc[i+INPUT_LEN] for i in range(n_win)]
X_tr, Y_tr = X[:train_end], Y[:train_end]
X_te, Y_te = X[split:], Y[split:]
dates_te = dates[split:]

Xa = np.array([[[v] for v in y_tot[i:i+INPUT_LEN]] for i in range(n_win)])
Xa_tr, Xa_te = Xa[:train_end], Xa[split:]

scaler_X = StandardScaler(); scaler_Y = StandardScaler()
X_tr_s = scaler_X.fit_transform(X_tr.reshape(-1,4)).reshape(X_tr.shape)
X_te_s = scaler_X.transform(X_te.reshape(-1,4)).reshape(X_te.shape)
Y_tr_s = scaler_Y.fit_transform(Y_tr)
Y_te_s = scaler_Y.transform(Y_te)
scaler_Xa = StandardScaler()
Xa_tr_s = scaler_Xa.fit_transform(Xa_tr.reshape(-1,1)).reshape(Xa_tr.shape)
Xa_te_s = scaler_Xa.transform(Xa_te.reshape(-1,1)).reshape(Xa_te.shape)
print(f"windows={n_win} train={train_end} (boundary-corrected) test={n_win-split}", flush=True)

# ===== Metrics =====
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

# ===== Deterministic baselines =====
preds_persist = np.array([[y_tot[split+i+INPUT_LEN-1]]*OUTPUT_LEN
                          for i in range(len(X_te))])
preds_snaive = np.array([[y_tot[split+i+INPUT_LEN+s-52] for s in range(OUTPUT_LEN)]
                         for i in range(len(X_te))])
mase_scale = mae(Y_te, preds_snaive)
print(f"Seasonal-naive pooled MAE (MASE scale) = {mase_scale:.4f}", flush=True)

# ===== ARIMA =====
print("\nTraining ARIMA...", flush=True)
y_seq = df[target].values
n_tr = int(len(y_seq) * TRAIN_RATIO)
try:
    arima_mod = sm.tsa.statespace.SARIMAX(y_seq[:n_tr], order=(1,1,1),
                                          seasonal_order=(1,1,1,52),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
    arima_res = arima_mod.fit(disp=False)
    preds_arima = []
    for w in range(len(Y_te)):
        hist_end = (split + w) + INPUT_LEN
        if w == 0:
            for obs_idx in range(n_tr, hist_end):
                arima_res = arima_res.append(np.array([y_seq[obs_idx]]), refit=False)
        else:
            arima_res = arima_res.append(np.array([y_seq[hist_end - 1]]), refit=False)
        fc = arima_res.get_forecast(steps=OUTPUT_LEN)
        preds_arima.append(np.asarray(fc.predicted_mean))
    preds_arima = np.array(preds_arima)
    Y_te_a = Y_te[:len(preds_arima)]
    print("ARIMA OK:", evaluate(Y_te_a, preds_arima), flush=True)
except Exception as e:
    print("ARIMA error:", e, flush=True)
    preds_arima = np.zeros_like(Y_te); Y_te_a = Y_te

# ===== XGBoost =====
print("Training XGBoost...", flush=True)
X_tr_xgb = X_tr.reshape(len(X_tr), -1)
X_te_xgb = X_te.reshape(len(X_te), -1)
preds_xgb = np.zeros_like(Y_te)
for step in range(OUTPUT_LEN):
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             objective="reg:squarederror", random_state=42, n_jobs=4,
                             reg_lambda=1.0, reg_alpha=0.0)
    model.fit(X_tr_xgb, Y_tr[:, step])
    preds_xgb[:, step] = model.predict(X_te_xgb)
print("XGBoost:", evaluate(Y_te, preds_xgb), flush=True)

# ===== PyTorch models =====
class PlainLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden=64, out_steps=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),
                                nn.Dropout(dropout),nn.Linear(hidden,out_steps))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]), None

class VSN(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.weight_net = nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),
                                        nn.Linear(hidden,input_dim))
        self.transform = nn.Linear(input_dim, hidden)
    def forward(self, x):
        ctx = x.mean(dim=1)
        w = torch.softmax(self.weight_net(ctx), dim=-1)
        return self.transform(x * w.unsqueeze(1)), w

class MSTFT(nn.Module):
    def __init__(self, input_dim=4, hidden=64, n_heads=4, out_steps=4, dropout=0.1):
        super().__init__()
        self.vsn = VSN(input_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),
                                nn.Dropout(dropout),nn.Linear(hidden,out_steps))
    def forward(self, x):
        sel, w = self.vsn(x)
        l, _ = self.lstm(sel)
        a, _ = self.attn(l, l, l)
        out = self.norm(l + a)
        return self.fc(out[:, -1, :]), w

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

def online_predict_single_step(model, Xte, Yte_s, online_epochs=3, lr=3e-4, wd=1e-5):
    """M1(a): update loss uses ONLY the first-step target - the single value
    actually observed before the next prediction."""
    mdl = copy.deepcopy(model)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    preds = []
    Xte_t = torch.FloatTensor(Xte); Yte_t = torch.FloatTensor(Yte_s)
    for i in range(len(Xte)):
        mdl.eval()
        with torch.no_grad():
            p, _ = mdl(Xte_t[i:i+1])
            preds.append(p.numpy()[0])
        mdl.train()
        for _ in range(online_epochs):
            opt.zero_grad()
            p, _ = mdl(Xte_t[i:i+1])
            loss = crit(p[:, :1], Yte_t[i:i+1, :1])
            loss.backward(); opt.step()
    return np.array(preds)

def online_predict_delayed(model, Xte, Yte_s, online_epochs=3, lr=3e-4, wd=1e-5,
                           lag=None):
    """M1(b): before predicting window i, fine-tune on window i-lag's FULL
    4-step target. Window i-lag's targets are fully observed at that moment
    (targets of window j span weeks split+j+12..split+j+15; observed iff
    j <= i-4), so no future information is used, while the full multi-step
    training signal is preserved."""
    if lag is None: lag = OUTPUT_LEN
    mdl = copy.deepcopy(model)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    preds = []
    Xte_t = torch.FloatTensor(Xte); Yte_t = torch.FloatTensor(Yte_s)
    for i in range(len(Xte)):
        mdl.eval()
        with torch.no_grad():
            p, _ = mdl(Xte_t[i:i+1])
            preds.append(p.numpy()[0])
        j = i - lag
        if j >= 0:
            mdl.train()
            for _ in range(online_epochs):
                opt.zero_grad()
                p, _ = mdl(Xte_t[j:j+1])
                loss = crit(p, Yte_t[j:j+1])
                loss.backward(); opt.step()
    return np.array(preds)

def periodic_predict(seed, every=13, epochs=100):
    """M3: full retraining on the expanding window every `every` windows,
    using only fully observed windows (k <= i - OUTPUT_LEN)."""
    np.random.seed(seed); torch.manual_seed(seed)
    model = train_base(PlainLSTM(), X_tr_s, Y_tr_s, epochs=200)
    preds = []
    for i in range(len(X_te_s)):
        if i > 0 and i % every == 0:
            k = i - OUTPUT_LEN
            X_all = np.concatenate([X_tr_s, X_te_s[:k]])
            Y_all = np.concatenate([Y_tr_s, Y_te_s[:k]])
            model = train_base(PlainLSTM(), X_all, Y_all, epochs=epochs)
        model.eval()
        with torch.no_grad():
            p, _ = model(torch.FloatTensor(X_te_s[i:i+1]))
            preds.append(p.numpy()[0])
    return np.array(preds)

# ===== 5-seed runs =====
DET = {"Persistence": preds_persist, "Seasonal naive": preds_snaive,
       "ARIMA": preds_arima, "XGBoost": preds_xgb}
SEED_MODELS = ["Static LSTM",
               "Online LSTM (single-step update)",
               "Online LSTM (delayed update)",
               "Periodic-retrain LSTM",
               "Static MS-TFT",
               "Online MS-TFT (delayed update)",
               "Agg-input Static LSTM",
               "Agg-input Online LSTM (delayed update)",
               "Delayed online (1 step)",
               "Delayed online (5 steps)"]
seed_preds = {m: [] for m in SEED_MODELS}
preds42 = {}

print("\nRunning 5 seeds...", flush=True)
for seed in SEEDS:
    np.random.seed(seed); torch.manual_seed(seed)
    print(f"\n--- Seed {seed} ({time.time()-t0:.0f}s) ---", flush=True)

    m1 = train_base(PlainLSTM(), X_tr_s, Y_tr_s, epochs=200)
    p1 = scaler_Y.inverse_transform(m1(torch.FloatTensor(X_te_s))[0].detach().numpy())
    seed_preds["Static LSTM"].append(p1)
    print("Static LSTM:", evaluate(Y_te, p1), flush=True)

    m2 = train_base(PlainLSTM(), X_tr_s, Y_tr_s, epochs=200)
    p2 = scaler_Y.inverse_transform(online_predict_single_step(m2, X_te_s, Y_te_s))
    seed_preds["Online LSTM (single-step update)"].append(p2)
    print("Online single-step:", evaluate(Y_te, p2), flush=True)

    m2b = train_base(PlainLSTM(), X_tr_s, Y_tr_s, epochs=200)
    p2b = scaler_Y.inverse_transform(online_predict_delayed(m2b, X_te_s, Y_te_s))
    seed_preds["Online LSTM (delayed update)"].append(p2b)
    print("Online delayed:", evaluate(Y_te, p2b), flush=True)

    p5 = scaler_Y.inverse_transform(periodic_predict(seed, every=13, epochs=100))
    seed_preds["Periodic-retrain LSTM"].append(p5)
    print("Periodic-retrain:", evaluate(Y_te, p5), flush=True)

    m3 = train_base(MSTFT(), X_tr_s, Y_tr_s, epochs=200)
    p3 = scaler_Y.inverse_transform(m3(torch.FloatTensor(X_te_s))[0].detach().numpy())
    seed_preds["Static MS-TFT"].append(p3)
    print("Static MS-TFT:", evaluate(Y_te, p3), flush=True)

    m4 = train_base(MSTFT(), X_tr_s, Y_tr_s, epochs=200)
    p4 = scaler_Y.inverse_transform(online_predict_delayed(m4, X_te_s, Y_te_s))
    seed_preds["Online MS-TFT (delayed update)"].append(p4)
    print("Online MS-TFT (delayed):", evaluate(Y_te, p4), flush=True)

    ma1 = train_base(PlainLSTM(input_dim=1), Xa_tr_s, Y_tr_s, epochs=200)
    pa1 = scaler_Y.inverse_transform(ma1(torch.FloatTensor(Xa_te_s))[0].detach().numpy())
    seed_preds["Agg-input Static LSTM"].append(pa1)
    ma2 = train_base(PlainLSTM(input_dim=1), Xa_tr_s, Y_tr_s, epochs=200)
    pa2 = scaler_Y.inverse_transform(online_predict_delayed(ma2, Xa_te_s, Y_te_s))
    seed_preds["Agg-input Online LSTM (delayed update)"].append(pa2)
    print("Agg Static:", evaluate(Y_te, pa1), "| Agg Online (delayed):", evaluate(Y_te, pa2), flush=True)

    m6a = train_base(PlainLSTM(), X_tr_s, Y_tr_s, epochs=200)
    p6a = scaler_Y.inverse_transform(online_predict_delayed(m6a, X_te_s, Y_te_s, online_epochs=1))
    seed_preds["Delayed online (1 step)"].append(p6a)
    m6b = train_base(PlainLSTM(), X_tr_s, Y_tr_s, epochs=200)
    p6b = scaler_Y.inverse_transform(online_predict_delayed(m6b, X_te_s, Y_te_s, online_epochs=5))
    seed_preds["Delayed online (5 steps)"].append(p6b)
    print("Sensitivity delayed 1-step:", evaluate(Y_te, p6a), "| 5-step:", evaluate(Y_te, p6b), flush=True)

    if seed == 42:
        preds42 = {"Static LSTM": p1, "Online single-step": p2,
                   "Online delayed": p2b, "Periodic-retrain LSTM": p5,
                   "Static MS-TFT": p3, "Online MS-TFT delayed": p4,
                   "Agg Static": pa1, "Agg Online delayed": pa2,
                   "ARIMA": preds_arima, "XGBoost": preds_xgb}

# save per-seed predictions for post-hoc analysis
np.savez(TAB + "seed_predictions.npz",
         **{m.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"):
            np.array(seed_preds[m]) for m in SEED_MODELS},
         Y_te=Y_te, Y_te_a=Y_te_a, preds_arima=preds_arima,
         preds_xgb=preds_xgb, preds_persist=preds_persist,
         preds_snaive=preds_snaive, mase_scale=mase_scale)
print("Saved seed_predictions.npz", flush=True)

# ===== Aggregation =====
ALL_MODELS = list(DET.keys()) + SEED_MODELS
def yt_for(name): return Y_te_a if name == "ARIMA" else Y_te

overall = {}
for name in ALL_MODELS:
    if name in DET:
        r = evaluate(yt_for(name), DET[name])
        overall[name] = {k: (float(v), 0.0) for k, v in r.items()}
    else:
        rs = [evaluate(Y_te, p) for p in seed_preds[name]]
        overall[name] = {k: (float(np.mean([r[k] for r in rs])),
                             float(np.std([r[k] for r in rs]))) for k in rs[0]}

by_step = {}
for name in ALL_MODELS:
    if name in DET:
        by_step[name] = {f"t+{s+1}": {k: (float(v), 0.0) for k, v in
                          evaluate(yt_for(name)[:,s], DET[name][:,s]).items()}
                         for s in range(OUTPUT_LEN)}
    else:
        per = [{f"t+{s+1}": evaluate(Y_te[:,s], p[:,s]) for s in range(OUTPUT_LEN)}
               for p in seed_preds[name]]
        by_step[name] = {h: {k: (float(np.mean([x[h][k] for x in per])),
                                 float(np.std([x[h][k] for x in per])))
                             for k in per[0][h]} for h in per[0]}

iso_weeks = df["iso_week"].values[INPUT_LEN + split : INPUT_LEN + split + len(Y_te)]
m_w = ((iso_weeks >= 48) | (iso_weeks <= 10)).astype(bool)
m_s = ((iso_weeks >= 22) & (iso_weeks <= 34)).astype(bool)
season_res = {}
for name in ALL_MODELS:
    if name in DET:
        pr = DET[name]
        season_res[name] = {se: {k: (float(v), 0.0) for k, v in
                            evaluate(yt_for(name)[mask][:,0], pr[mask][:,0]).items()}
                            for se, mask in [("winter", m_w), ("summer", m_s)]}
    else:
        per = [{se: evaluate(Y_te[mask][:,0], p[mask][:,0])
                for se, mask in [("winter", m_w), ("summer", m_s)]}
               for p in seed_preds[name]]
        season_res[name] = {se: {k: (float(np.mean([x[se][k] for x in per])),
                                     float(np.std([x[se][k] for x in per])))
                                 for k in per[0][se]} for se in per[0]}

# ===== DM tests =====
def errors_of(name):
    if name in DET:
        return [(DET[name] - yt_for(name)).flatten()]
    return [(p - Y_te).flatten() for p in seed_preds[name]]

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

dm = {}
e_static = errors_of("Static LSTM")
e_single = errors_of("Online LSTM (single-step update)")
e_delay  = errors_of("Online LSTM (delayed update)")
e_period = errors_of("Periodic-retrain LSTM")
e_aggon  = errors_of("Agg-input Online LSTM (delayed update)")
e_arima  = errors_of("ARIMA")
e_snaive = errors_of("Seasonal naive")
dm["Delayed online vs Static LSTM"] = dm_over_seeds(e_delay, e_static)
dm["Delayed online vs single-step online"] = dm_over_seeds(e_delay, e_single)
dm["Delayed online vs ARIMA"] = dm_over_seeds(e_delay, e_arima)
dm["Delayed online vs Periodic-retrain"] = dm_over_seeds(e_delay, e_period)
dm["Delayed online vs Seasonal naive"] = dm_over_seeds(e_delay, e_snaive)
dm["Multi-subtype vs Aggregate (delayed online)"] = dm_over_seeds(e_delay, e_aggon)
print("\nDM tests:", json.dumps(dm, indent=1), flush=True)

# ===== results.json =====
results = {
    "mase_scale_snaive": round(mase_scale, 4),
    "overall_5seed": {n: {k: [round(v[0],4), round(v[1],4)] for k, v in d.items()}
                      for n, d in overall.items()},
    "by_horizon_5seed": {n: {h: {k: [round(v[0],4), round(v[1],4)] for k, v in d.items()}
                             for h, d in hd.items()} for n, hd in by_step.items()},
    "season_t1_5seed": {n: {se: {k: [round(v[0],4), round(v[1],4)] for k, v in d.items()}
                            for se, d in sd.items()} for n, sd in season_res.items()},
    "dm_tests": dm,
}
with open(TAB+"results.json","w",encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print("Saved results.json", flush=True)

# ===== Tables =====
def fmt(pair, nd=4):
    return f"{pair[0]:.{nd}f}±{pair[1]:.{nd}f}" if pair[1] > 0 else f"{pair[0]:.{nd}f}"

t1_rows = []
for name in ALL_MODELS:
    o = overall[name]
    t1_rows.append({"Model": name,
        "MAE (mean±SD)": fmt(o["MAE"]),
        "RMSE (mean±SD)": fmt(o["RMSE"]),
        "MASE": f"{o['MAE'][0]/mase_scale:.2f}",
        "R (mean±SD)": fmt(o["R"], 3)})
pd.DataFrame(t1_rows).to_csv(TAB+"Table1_model_comparison.csv", index=False)

t2_rows = []
for step in range(OUTPUT_LEN):
    h = f"t+{step+1}"
    for m in ALL_MODELS:
        d = by_step[m][h]
        t2_rows.append({"Horizon": h, "Model": m,
                        "MAE": fmt(d["MAE"]), "RMSE": fmt(d["RMSE"]),
                        "R": fmt(d["R"], 3)})
pd.DataFrame(t2_rows).to_csv(TAB+"Table2_by_horizon.csv", index=False)

t3_rows = []
for season in ["winter","summer"]:
    for m in ALL_MODELS:
        d = season_res[m][season]
        t3_rows.append({"Season": season.capitalize(), "Model": m,
                        "MAE": fmt(d["MAE"]), "RMSE": fmt(d["RMSE"]),
                        "R": fmt(d["R"], 3)})
pd.DataFrame(t3_rows).to_csv(TAB+"Table3_season.csv", index=False)

t4_rows = []
for m in ["Static LSTM","Online LSTM (single-step update)","Online LSTM (delayed update)",
          "Periodic-retrain LSTM","Static MS-TFT","Online MS-TFT (delayed update)",
          "Agg-input Static LSTM","Agg-input Online LSTM (delayed update)",
          "Delayed online (1 step)","Delayed online (5 steps)"]:
    o = overall[m]
    t4_rows.append({"Model": m,
        "MAE (mean±SD)": fmt(o["MAE"]), "RMSE (mean±SD)": fmt(o["RMSE"]),
        "R (mean±SD)": fmt(o["R"], 3)})
pd.DataFrame(t4_rows).to_csv(TAB+"Table4_ablation.csv", index=False)

t5_rows = []
for k, v in dm.items():
    t5_rows.append({"Comparison": k, "Mean DM statistic": v["mean_stat"],
                    "Mean p-value": v["mean_p"],
                    "Seeds significant (p<0.05)": f"{v['n_sig']}/5"})
pd.DataFrame(t5_rows).to_csv(TAB+"Table5_dm_tests.csv", index=False)
print("Saved Tables 1-5.", flush=True)

# ===== Figures =====
plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.linewidth":0.8,"axes.edgecolor":"#222222",
    "xtick.color":"#222222","ytick.color":"#222222",
    "axes.labelcolor":"#222222","text.color":"#222222"})

# Figure 2: online (delayed) vs static per-sample error, original scale
fig, ax = plt.subplots(figsize=(8, 4.5))
prog_on = np.abs(preds42["Online delayed"][:,0] - Y_te[:,0])
prog_st = np.abs(preds42["Static LSTM"][:,0] - Y_te[:,0])
roll_on = pd.Series(prog_on).rolling(10, min_periods=1).mean()
roll_st = pd.Series(prog_st).rolling(10, min_periods=1).mean()
ax.plot(range(len(prog_on)), prog_on, color="#CCCCCC", lw=0.4, alpha=0.5)
ax.plot(range(len(prog_st)), prog_st, color="#E8E8E8", lw=0.4, alpha=0.5)
ax.plot(range(len(roll_on)), roll_on, color="#C0504D", lw=1.8, label="Online LSTM, delayed update (10-sample rolling mean)")
ax.plot(range(len(roll_st)), roll_st, color="#4F81BD", lw=1.8, label="Static LSTM (10-sample rolling mean)")
ax.set_xlabel("Test sample index (chronological)")
ax.set_ylabel("Absolute error, t+1 horizon (percentage points)")
ax.set_title("Walk-forward adaptation: online (delayed update) vs static LSTM (seed 42)")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG+"Figure2_online_adaptation.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure2_online_adaptation.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 2 saved.", flush=True)

# Figure 4: prediction vs observed + ARIMA
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
for s, ax in enumerate(axes.flat):
    ax.plot(dates_te, Y_te[:, s], color="#333333", lw=1.2, label="Observed")
    ax.plot(dates_te, preds42["Online delayed"][:, s], color="#C0504D", lw=1.0, label="Online LSTM (delayed)")
    ax.plot(dates_te, preds42["Static LSTM"][:, s], color="#4F81BD", lw=0.8,
            label="Static LSTM", alpha=0.7, linestyle="--")
    ax.plot(dates_te, preds_arima[:, s], color="#888888", lw=0.8,
            label="ARIMA (rolling)", alpha=0.7, linestyle=":")
    ax.set_title(f"t+{s+1} week horizon")
    ax.set_ylabel("Positivity rate (%)")
    ax.spines[["top","right"]].set_visible(False)
    if s == 0: ax.legend(frameon=False, fontsize=7)
fig.autofmt_xdate()
fig.savefig(FIG+"Figure4_prediction_vs_observed.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure4_prediction_vs_observed.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 4 saved.", flush=True)

# Figure 5: ablation + controls
fig, ax = plt.subplots(figsize=(8.5, 4.5))
abl_models = ["Static LSTM","Online LSTM (single-step update)","Online LSTM (delayed update)",
              "Periodic-retrain LSTM","Static MS-TFT","Online MS-TFT (delayed update)",
              "Agg-input Static LSTM","Agg-input Online LSTM (delayed update)"]
abl_mae = [overall[m]["MAE"][0] for m in abl_models]
abl_mae_err = [overall[m]["MAE"][1] for m in abl_models]
abl_r = [overall[m]["R"][0] for m in abl_models]
x = np.arange(len(abl_models)); w = 0.35
ax.bar(x - w/2, abl_mae, w, yerr=abl_mae_err, color="#4F81BD", label="MAE", capsize=3)
ax2 = ax.twinx()
ax2.bar(x + w/2, abl_r, w, color="#C0504D", label="R")
ax.set_xticks(x); ax.set_xticklabels(
    [m.replace(" (", "\n(") for m in abl_models], fontsize=6.5)
ax.set_ylabel("MAE", color="#4F81BD"); ax2.set_ylabel("R", color="#C0504D")
ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
ax.set_title("Ablation and controls (5-seed mean±SD)")
fig.savefig(FIG+"Figure5_ablation.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure5_ablation.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 5 saved.", flush=True)

# Figure 6: horizon-stratified R (5-seed mean)
fig, ax = plt.subplots(figsize=(8, 4.5))
horizons = [f"t+{s+1}" for s in range(OUTPUT_LEN)]
models_to_plot = ["Seasonal naive","ARIMA","XGBoost","Static LSTM",
                  "Online LSTM (delayed update)"]
colors = ["#BBBBBB", "#888888", "#ED7D31", "#4F81BD", "#C0504D"]
for m, c in zip(models_to_plot, colors):
    rs = [by_step[m][h]["R"][0] for h in horizons]
    ax.plot(horizons, rs, marker="o", color=c, lw=1.5, label=m)
ax.set_ylabel("Pearson R"); ax.set_xlabel("Forecast horizon")
ax.set_title("Correlation by forecast horizon (5-seed mean)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(FIG+"Figure6_horizon_r.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure6_horizon_r.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 6 saved.", flush=True)

# Figure 7: winter vs summer (5-seed mean)
fig, ax = plt.subplots(figsize=(8, 4.5))
models_s = ["Seasonal naive","ARIMA","XGBoost","Static LSTM",
            "Online LSTM (delayed update)"]
x = np.arange(len(models_s)); w = 0.35
winter_mae = [season_res[m]["winter"]["MAE"][0] for m in models_s]
summer_mae = [season_res[m]["summer"]["MAE"][0] for m in models_s]
ax.bar(x - w/2, winter_mae, w, color="#4F81BD", label="Winter (weeks 48-10)")
ax.bar(x + w/2, summer_mae, w, color="#ED7D31", label="Summer (weeks 22-34)")
ax.set_xticks(x); ax.set_xticklabels(models_s, fontsize=7.5, rotation=15)
ax.set_ylabel("MAE, t+1 horizon"); ax.set_title("Winter vs summer performance (5-seed mean)")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(FIG+"Figure7_season.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG+"Figure7_season.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 7 saved.", flush=True)

print(f"\nAll done in {time.time()-t0:.0f}s")
