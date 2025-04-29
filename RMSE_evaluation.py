"""
Evaluate forecasting performance of PBM surrogates, GPR, Baseline RNN, and PI-RNN.

This script:
  1. Loads and preprocesses battery cycling & RPT data.
  2. Trains a RandomForest surrogate on physics-based model (PBM) capacity-drop data.
  3. Builds and trains:
       • PI-RNN (Physics-Informed RNN) for multi-step forecasting (horizon=10)
       • Baseline multi-step RNN
  4. Evaluates **single-step** predictions and plots 2*2 scatter plots with MAE/RMSE.
  5. Evaluates **multi-step** RMSE (horizons 2-10) for all four methods and plots a bar chart.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_utils import (
    load_pbm_surrogate,
    load_battery_data,
    make_sequences,
    PBM_FEATURES
)
from models import PBMSurrogate, MultiStepPIRNN, BaselineMultiStepRNN, GPRBaseline

# —————————————————————————————
# 0. Styling & reproducibility
# —————————————————————————————
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 14

seed = 40
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# for our scatter plots
marker_size = 100
tick_range = np.arange(0.6, 1.3, 0.1)

# —————————————————————————————
# 1. Train PBM surrogate
# —————————————————————————————
rf_model, scaler_sim, sim_df = load_pbm_surrogate(seed=seed)

# wrap into the full PBMSurrogate class (for single‐ and multi‐step)
# exclude "Capacity" from features during single‐step
drop_features = [f for f in PBM_FEATURES if f != 'Capacity']
pbm = PBMSurrogate(
    features=drop_features,
    capacity_target='Capacity',
    drop_target='Capacity_Drop_Ah',
    n_estimators=50,
    random_state=seed
)
# fit capacity‐drop model
pbm.fit_capacity(sim_df)
# fit drop model + multi‐horizon surrogates
pbm.fit_drop(sim_df)
for h in range(2, 11):
    pbm.fit_horizon(sim_df, h)

# —————————————————————————————
# 2. Load & preprocess battery data
# —————————————————————————————
X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df = \
    load_battery_data(seed=seed)

# build sequences of length 10 ▶ scenario 3 training
horizon = 10
X_tr_seq, y_tr_seq = make_sequences(X_train_s, y_train, horizon)
X_va_seq, y_va_seq = make_sequences(X_val_s,   y_val,   horizon)
X_te_seq, y_te_seq = make_sequences(X_test_s,  y_test,  horizon)

# to tensors
X_tr = torch.tensor(X_tr_seq, dtype=torch.float32)
y_tr = torch.tensor(y_tr_seq, dtype=torch.float32)
X_va = torch.tensor(X_va_seq, dtype=torch.float32)
y_va = torch.tensor(y_va_seq, dtype=torch.float32)
X_te = torch.tensor(X_te_seq, dtype=torch.float32)
y_te = torch.tensor(y_te_seq, dtype=torch.float32)

# —————————————————————————————
# 3. Scenario 3: Train max‐horizon PI-RNN & Baseline RNN
# —————————————————————————————
input_size  = X_tr.shape[-1] + 1   # features + current capacity
hidden_size = 50
num_epochs  = 2500
patience    = 50
criterion   = nn.MSELoss()

# PI-RNN
pi_rnn = MultiStepPIRNN(input_size, hidden_size, rf_model)
opt_pi = optim.Adam(pi_rnn.parameters(), lr=1e-3)

best_val, no_imp = float('inf'), 0
for ep in range(1, num_epochs+1):
    pi_rnn.train(); opt_pi.zero_grad()
    seed_cap = y_tr[:, [0]]
    preds = pi_rnn(X_tr, seed_cap, forecast_steps=horizon)
    loss  = criterion(preds, y_tr)
    loss.backward(); opt_pi.step()

    pi_rnn.eval()
    with torch.no_grad():
        vseed = y_va[:, [0]]
        vpred = pi_rnn(X_va, vseed, forecast_steps=horizon)
        vloss = criterion(vpred, y_va)

    if vloss < best_val:
        best_val, no_imp = vloss, 0
    else:
        no_imp += 1
        if no_imp >= patience:
            print(f"[PI-RNN] early stop @ epoch {ep}")
            break

# Baseline RNN
base_rnn = BaselineMultiStepRNN(input_size, hidden_size)
opt_b   = optim.Adam(base_rnn.parameters(), lr=1e-3)

best_vb, no_impb = float('inf'), 0
for ep in range(1, num_epochs+1):
    base_rnn.train(); opt_b.zero_grad()
    bpreds = base_rnn(X_tr, seed_cap, forecast_steps=horizon)
    bloss  = criterion(bpreds, y_tr)
    bloss.backward(); opt_b.step()

    base_rnn.eval()
    with torch.no_grad():
        vbpred = base_rnn(X_va, vseed, forecast_steps=horizon)
        vbloss = criterion(vbpred, y_va)

    if vbloss < best_vb:
        best_vb, no_impb = vbloss, 0
    else:
        no_impb += 1
        if no_impb >= patience:
            print(f"[Baseline RNN] early stop @ epoch {ep}")
            break

# —————————————————————————————
# 4. Single-step evaluation
# —————————————————————————————
def eval_single(model, X, y):
    model.eval()
    with torch.no_grad():
        seed_c = y[:, [0]]
        out    = model(X, seed_c, forecast_steps=horizon)
    return out[:,0].cpu().numpy()

# RNNs
rnn_pred   = eval_single(base_rnn, X_te, y_te)
pirnn_pred = eval_single(pi_rnn,   X_te, y_te)

# GPR baseline
model_gpr = GPRBaseline()
y_true_gpr, y_pred_gpr = [], []
for (g, c), grp in test_df.groupby(['Group','Cell']):
    model_gpr.fit(grp, (g,c), initial_points=6)
    yt, yp = model_gpr.predict(grp, (g,c), initial_points=6)
    y_true_gpr.extend(yt); y_pred_gpr.extend(yp)

# PBM surrogate single-step
y_true_pbm = test_df['Capacity'].values
y_pred_pbm = pbm.predict_capacity(test_df)

# compute MAE/RMSE
rmse_pbm, mae_pbm   = np.sqrt(mean_squared_error(y_true_pbm, y_pred_pbm)), mean_absolute_error(y_true_pbm, y_pred_pbm)
rmse_gpr, mae_gpr   = np.sqrt(mean_squared_error(y_true_gpr, y_pred_gpr)), mean_absolute_error(y_true_gpr, y_pred_gpr)
rmse_rnn, mae_rnn   = np.sqrt(mean_squared_error(y_te[:,0].numpy(), rnn_pred)), mean_absolute_error(y_te[:,0].numpy(), rnn_pred)
rmse_pirnn, mae_pirnn = np.sqrt(mean_squared_error(y_te[:,0].numpy(), pirnn_pred)), mean_absolute_error(y_te[:,0].numpy(), pirnn_pred)

# —————————————————————————————
# 5. Single-step plotting (2×2)
# —————————————————————————————
fig, axs = plt.subplots(2, 2, figsize=(15,9), dpi=300, constrained_layout=True)
models = [
    (y_true_pbm,   y_pred_pbm,   'PBM',           'lightgreen', 'o', rmse_pbm,   mae_pbm),
    (y_true_gpr,   y_pred_gpr,   'GPR',           'crimson',    '^', rmse_gpr,   mae_gpr),
    (y_te[:,0].numpy(), rnn_pred, 'Baseline RNN',  'orange',     's', rmse_rnn,   mae_rnn),
    (y_te[:,0].numpy(), pirnn_pred,'PI-RNN',        'green',      'D', rmse_pirnn, mae_pirnn)
]
for ax, (t, p, title, color, m, rmse, mae) in zip(axs.flat, models):
    ax.scatter(t, p, color=color, marker=m, s=marker_size, alpha=0.7)
    ax.plot([0.5,1.3],[0.5,1.3],'k--',lw=1.5)
    ax.set_title(title)
    ax.set_xlabel('True Capacity (Ah)')
    ax.set_ylabel('Predicted Capacity (Ah)')
    ax.set_xticks(tick_range); ax.set_yticks(tick_range)
    ax.set_xlim([0.55,1.25]);  ax.set_ylim([0.55,1.25])
    ax.text(1.05, 0.6, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
            fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.show()

# —————————————————————————————
# 6. Multi-step RMSE
# —————————————————————————————
horizons = list(range(2,11))
rmse_multi = defaultdict(list)

# PBM
for h in horizons:
    yt = y_true_pbm[h:]
    yp = pbm.predict_capacity_multi(test_df, steps=h)
    rmse_multi['PBM'].append(np.sqrt(mean_squared_error(yt, yp)))

# GPR
gpr2 = GPRBaseline(initial_points=9)
for (g,c), grp in test_df.groupby(['Group','Cell']):
    gpr2.fit(grp, (g,c), initial_points=9)
for h in horizons:
    yt_all, yp_all = [], []
    for (g,c), grp in test_df.groupby(['Group','Cell']):
        yt, yp = gpr2.predict_horizon(grp, (g,c), steps=h)
        yt_all.extend(yt); yp_all.extend(yp)
    rmse_multi['GPR'].append(np.sqrt(mean_squared_error(yt_all, yp_all)))

# RNN & PI-RNN
base_rnn.eval(); pi_rnn.eval()
with torch.no_grad():
    for h in horizons:
        b_out = base_rnn(X_te, y_te[:,:1], forecast_steps=h).cpu().numpy().flatten()
        p_out = pi_rnn(X_te,   y_te[:,:1], forecast_steps=h).cpu().numpy().flatten()
        t_true = y_te[:,:h].cpu().numpy().flatten()

        rmse_multi['RNN'].append(np.sqrt(mean_squared_error(t_true, b_out)))
        rmse_multi['PI-RNN'].append(np.sqrt(mean_squared_error(t_true, p_out)))

# bar plot
x = np.arange(len(horizons))
w = 0.2

plt.figure(figsize=(10,5), dpi=300)
plt.bar(x-1.5*w, rmse_multi['PBM'],    w, label='PBM',    color='lightgreen')
plt.bar(x-0.5*w, rmse_multi['GPR'],    w, label='GPR',    color='crimson')
plt.bar(x+0.5*w, rmse_multi['RNN'],    w, label='RNN',    color='orange')
plt.bar(x+1.5*w, rmse_multi['PI-RNN'], w, label='PI-RNN', color='green')

plt.xlabel('Forecasting Steps', fontsize=14)
plt.ylabel('RMSE (Ah)',            fontsize=14)
plt.xticks(x, horizons)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
