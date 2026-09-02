# -*- coding: utf-8 -*-
"""
Step 7: replay-based walk-forward fine-tuning (final due diligence for the
online-learning story). Experience replay is the standard remedy for noisy
single-sample online updates: at each window i, take 1 gradient step on the
last K fully-observed test windows (windows <= i-4). Leakage-free.
"""
import time, warnings, copy
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn

t0 = time.time()
TAB = "../tables/"
df = pd.read_csv(TAB + "main_series_2015_2025.csv", encoding="utf-8-sig", parse_dates=["week_start"])

INPUT_LEN, OUTPUT_LEN, TRAIN_RATIO = 12, 4, 0.70
SEEDS = [42, 123, 777, 2024, 8888]
features = ["rate_h3n2","rate_h1n1","rate_by","rate_bv"]
target = "rate_total"

y_tot = df[target].values
vals = df[features].values
n_win = len(df) - INPUT_LEN - OUTPUT_LEN + 1
split = int(n_win * TRAIN_RATIO)
X = np.array([vals[i:i+INPUT_LEN] for i in range(n_win)])
Y = np.array([y_tot[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN] for i in range(n_win)])
X_tr, Y_tr = X[:split], Y[:split]
X_te, Y_te = X[split:], Y[split:]
Xa = np.array([[[v] for v in y_tot[i:i+INPUT_LEN]] for i in range(n_win)])
Xa_tr, Xa_te = Xa[:split], Xa[split:]

scaler_X = StandardScaler(); scaler_Y = StandardScaler()
X_tr_s = scaler_X.fit_transform(X_tr.reshape(-1,4)).reshape(X_tr.shape)
X_te_s = scaler_X.transform(X_te.reshape(-1,4)).reshape(X_te.shape)
Y_tr_s = scaler_Y.fit_transform(Y_tr)
Y_te_s = scaler_Y.transform(Y_te)
scaler_Xa = StandardScaler()
Xa_tr_s = scaler_Xa.fit_transform(Xa_tr.reshape(-1,1)).reshape(Xa_tr.shape)
Xa_te_s = scaler_Xa.transform(Xa_te.reshape(-1,1)).reshape(Xa_te.shape)

class PlainLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden=64, out_steps=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),
                                nn.Dropout(dropout),nn.Linear(hidden,out_steps))
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

def mae(a,b): return float(np.mean(np.abs(a-b)))
def pearson(a,b): return float(np.corrcoef(np.asarray(a).flatten(), np.asarray(b).flatten())[0,1])

def online_predict_replay(model, Xte, Yte_s, K=26, steps=1, lr=3e-4, wd=1e-5):
    """At each window i: 1 gradient step on the last K fully-observed windows
    (indices <= i-OUTPUT_LEN). No future information."""
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
        j = i - OUTPUT_LEN
        if j >= 0:
            lo = max(0, j - K + 1)
            xb = Xte_t[lo:j+1]; yb = Yte_t[lo:j+1]
            mdl.train()
            for _ in range(steps):
                opt.zero_grad()
                p, _ = mdl(xb)
                loss = crit(p, yb)
                loss.backward(); opt.step()
    return np.array(preds)

CONFIGS = [
    ("Replay K=26, 1 step, lr 3e-4", dict(K=26, steps=1, lr=3e-4)),
    ("Replay K=52, 1 step, lr 3e-4", dict(K=52, steps=1, lr=3e-4)),
    ("Replay K=26, 1 step, lr 1e-4", dict(K=26, steps=1, lr=1e-4)),
]

for label, cfg in CONFIGS:
    print(f"\n===== {label} =====", flush=True)
    for inname, Xtr_s, Xtest_s, dim in [("multi-subtype", X_tr_s, X_te_s, 4),
                                          ("aggregate", Xa_tr_s, Xa_te_s, 1)]:
        maes, rs = [], []
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed)
            m = train_base(PlainLSTM(input_dim=dim), Xtr_s, Y_tr_s, epochs=200)
            p = scaler_Y.inverse_transform(online_predict_replay(m, Xtest_s, Y_te_s, **cfg))
            maes.append(mae(Y_te, p)); rs.append(pearson(Y_te, p))
        print(f"{inname}: MAE={np.mean(maes):.4f}±{np.std(maes):.4f}  "
              f"R={np.mean(rs):.3f}±{np.std(rs):.3f}", flush=True)

print(f"\nDone in {time.time()-t0:.0f}s")
