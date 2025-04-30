"""
Train and compare physics-informed RNN (PI-RNN) and baseline RNN across three forecasting scenarios:

  • Scenario 1: Fixed 7-step horizon forecasting  
  • Scenario 2: Recursive single-step forecasting  
  • Scenario 3: Maximum 10-step horizon forecasting  
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from data_utils import (
    PBM_SIM_PATHS,
    PBM_FEATURES,
    PBM_TARGET,
    BATTERY_FEATURES,
    BATTERY_TARGET,
    load_battery_data,
    make_sequences
)
from models import train_pbm_surrogate_for_PI_RNN, MultiStepPIRNN, BaselineMultiStepRNN

# —————————————————————————————
# 0. Styling & Reproducibility
# —————————————————————————————
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 18

seed = 40
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# —————————————————————————————
# 1. Train PBM surrogate (capacity-drop)
# —————————————————————————————
rf_model, scaler_sim = train_pbm_surrogate_for_PI_RNN(
    PBM_SIM_PATHS,
    PBM_FEATURES,
    PBM_TARGET,
    seed=seed
)

# —————————————————————————————
# 2. Load & preprocess battery data
# —————————————————————————————
X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df = \
    load_battery_data(seed=seed)
features = BATTERY_FEATURES
target   = BATTERY_TARGET

# —————————————————————————————
# 3. Build sequences for scenarios
# —————————————————————————————
h1, h2, h3 = 7, 1, 10
X_tr1, y_tr1 = make_sequences(X_train_s, y_train, h1)
X_va1, y_va1 = make_sequences(X_val_s,   y_val,   h1)
X_tr2, y_tr2 = make_sequences(X_train_s, y_train, h2)
X_va2, y_va2 = make_sequences(X_val_s,   y_val,   h2)
X_tr3, y_tr3 = make_sequences(X_train_s, y_train, h3)
X_va3, y_va3 = make_sequences(X_val_s,   y_val,   h3)

def T(x): return torch.tensor(x, dtype=torch.float32)
X_tr1_t, y_tr1_t = T(X_tr1), T(y_tr1)
X_va1_t, y_va1_t = T(X_va1), T(y_va1)
X_tr2_t, y_tr2_t = T(X_tr2), T(y_tr2)
X_va2_t, y_va2_t = T(X_va2), T(y_va2)
X_tr3_t, y_tr3_t = T(X_tr3), T(y_tr3)
X_va3_t, y_va3_t = T(X_va3), T(y_va3)

# —————————————————————————————
# 4. Scenario-training helper
# —————————————————————————————
def train_scenario(model, optimizer, Xtr, ytr, Xva, yva, horizon, name):
    best_val, no_imp = float('inf'), 0
    criterion = nn.MSELoss()
    for ep in range(1, 2501):
        model.train(); optimizer.zero_grad()
        seed_cap = ytr[:, :1]
        preds    = model(Xtr, seed_cap, forecast_steps=horizon)
        loss     = criterion(preds, ytr[:, :horizon])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            vpred = model(Xva, yva[:, :1], forecast_steps=horizon)
            vloss = criterion(vpred, yva[:, :horizon])

        if vloss < best_val:
            best_val, no_imp = vloss, 0
        else:
            no_imp += 1
            if no_imp >= 150:
                print(f"[{name}] early stop @ epoch {ep}")
                break

        if ep % 100 == 0:
            print(f"[{name}] Epoch {ep}: train={loss:.4f} val={vloss:.4f}")

# —————————————————————————————
# 5. Scenario 1: 7-step fixed-horizon
# —————————————————————————————
input_size  = len(features) + 1
hidden_size = 50

torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
pi1  = MultiStepPIRNN(input_size, hidden_size, rf_model)
opt1 = optim.Adam(pi1.parameters(), lr=1e-3)
train_scenario(pi1, opt1, X_tr1_t, y_tr1_t, X_va1_t, y_va1_t, h1, "PI-RNN S1")

torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
b1   = BaselineMultiStepRNN(input_size, hidden_size)
ob1  = optim.Adam(b1.parameters(), lr=1e-3)
train_scenario(b1,  ob1,  X_tr1_t, y_tr1_t, X_va1_t, y_va1_t, h1, "Base-RNN S1")

# —————————————————————————————
# 6. Scenario 2: recursive single-step
# —————————————————————————————
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
pi2  = MultiStepPIRNN(input_size, hidden_size, rf_model)
opt2 = optim.Adam(pi2.parameters(), lr=1e-3)
train_scenario(pi2, opt2, X_tr2_t, y_tr2_t, X_va2_t, y_va2_t, h2, "PI-RNN S2")

torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
b2   = BaselineMultiStepRNN(input_size, hidden_size)
ob2  = optim.Adam(b2.parameters(), lr=1e-3)
train_scenario(b2,  ob2,  X_tr2_t, y_tr2_t, X_va2_t, y_va2_t, h2, "Base-RNN S2")

# —————————————————————————————
# 7. Scenario 3: 10-step max-horizon
# —————————————————————————————
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
pi3  = MultiStepPIRNN(input_size, hidden_size, rf_model)
opt3 = optim.Adam(pi3.parameters(), lr=1e-3)
train_scenario(pi3, opt3, X_tr3_t, y_tr3_t, X_va3_t, y_va3_t, h3, "PI-RNN S3")

torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
b3   = BaselineMultiStepRNN(input_size, hidden_size)
ob3  = optim.Adam(b3.parameters(), lr=1e-3)
train_scenario(b3,  ob3,  X_tr3_t, y_tr3_t, X_va3_t, y_va3_t, h3, "Base-RNN S3")

# —————————————————————————————
# 7.1 Save trained S3 models
# —————————————————————————————
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)
torch.save(pi3.state_dict(), f"{model_dir}/pi3_scenario3.pth")
torch.save(b3.state_dict(),  f"{model_dir}/b3_scenario3.pth")
print(f"Saved PI-RNN S3 and Baseline S3 to {model_dir}")

# —————————————————————————————
# 8. Visualization + fine-tuning on first 5 points
# —————————————————————————————
def visualize_all_scenarios(
    forecast_rpt, forecast_steps, scaler, features, target,
    Group='G3', Cell='C1', return_predictions=False,
    fine_tune=False, fine_tune_epochs=20
):
    # pick original models or clones
    if fine_tune:
        m1, m2, m3 = deepcopy(pi1), deepcopy(pi2), deepcopy(pi3)
        b1_, b2_, b3_ = deepcopy(b1), deepcopy(b2), deepcopy(b3)
    else:
        m1, m2, m3 = pi1, pi2, pi3
        b1_, b2_, b3_ = b1, b2, b3

    df = test_df[(test_df['Group']==Group)&(test_df['Cell']==Cell)].copy()
    df.sort_values('RPT Number', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if 23 not in df['RPT Number'].values and {22,24}.issubset(df['RPT Number'].values):
        v22 = df.loc[df['RPT Number']==22, target].item()
        v24 = df.loc[df['RPT Number']==24, target].item()
        extra = pd.DataFrame({'RPT Number':[23], target:[(v22+v24)/2]})
        df = pd.concat([df, extra], ignore_index=True)
        df.sort_values('RPT Number', inplace=True)
        df.reset_index(drop=True, inplace=True)

    avail = df[df['RPT Number'] <= forecast_rpt]
    fut   = df[df['RPT Number'] >  forecast_rpt]
    fw    = fut.iloc[:forecast_steps]
    n     = len(fw)

    # fine-tune on first 5 available points
    if fine_tune and len(avail) >= 5:
        raw_feats = avail[features].values[:5]
        raw_tgts  = avail[target].values[:5]
        scaled_5  = scaler.transform(raw_feats)

        Xm, ym = make_sequences(scaled_5, raw_tgts, 5)
        Xs, ys = make_sequences(scaled_5, raw_tgts, 1)
        Xm_t, ym_t = torch.tensor(Xm, dtype=torch.float32), torch.tensor(ym, dtype=torch.float32)
        Xs_t, ys_t = torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

        opts = [
            optim.Adam(m1.parameters(), lr=1e-3),
            optim.Adam(b1_.parameters(), lr=1e-3),
            optim.Adam(m2.parameters(), lr=1e-3),
            optim.Adam(b2_.parameters(), lr=1e-3),
            optim.Adam(m3.parameters(), lr=1e-3),
            optim.Adam(b3_.parameters(), lr=1e-3),
        ]
        loss_fn = nn.MSELoss()

        m1.train(); b1_.train()
        m2.train(); b2_.train()
        m3.train(); b3_.train()
        for _ in range(fine_tune_epochs):
            # S1: 5-step
            opts[0].zero_grad()
            p1 = m1(Xm_t, ym_t[:, :1], forecast_steps=5)
            l1 = loss_fn(p1, ym_t)
            l1.backward(); opts[0].step()

            opts[1].zero_grad()
            bp1 = b1_(Xm_t, ym_t[:, :1], forecast_steps=5)
            lb1 = loss_fn(bp1, ym_t)
            lb1.backward(); opts[1].step()

            # S2: 1-step
            opts[2].zero_grad()
            p2 = m2(Xs_t, ys_t[:, :1], forecast_steps=1)
            l2 = loss_fn(p2, ys_t)
            l2.backward(); opts[2].step()

            opts[3].zero_grad()
            bp2 = b2_(Xs_t, ys_t[:, :1], forecast_steps=1)
            lb2 = loss_fn(bp2, ys_t)
            lb2.backward(); opts[3].step()

            # S3: 5-step
            opts[4].zero_grad()
            p3 = m3(Xm_t, ym_t[:, :1], forecast_steps=5)
            l3 = loss_fn(p3, ym_t)
            l3.backward(); opts[4].step()

            opts[5].zero_grad()
            bp3 = b3_(Xm_t, ym_t[:, :1], forecast_steps=5)
            lb3 = loss_fn(bp3, ym_t)
            lb3.backward(); opts[5].step()

        m1.eval(); b1_.eval()
        m2.eval(); b2_.eval()
        m3.eval(); b3_.eval()

    # prepare future window
    Xf_t = torch.tensor(scaler.transform(fw[features].fillna(0)),
                       dtype=torch.float32).unsqueeze(0)
    seed_t = torch.tensor([[avail[target].iloc[-1]]],
                          dtype=torch.float32)

    with torch.no_grad():
        pi1_p = m1(Xf_t, seed_t, forecast_steps=n).squeeze(0).cpu().numpy()
        b1_p  = b1_(Xf_t, seed_t, forecast_steps=n).squeeze(0).cpu().numpy()
        pi2_p = m2(Xf_t, seed_t, forecast_steps=n).squeeze(0).cpu().numpy()
        b2_p  = b2_(Xf_t, seed_t, forecast_steps=n).squeeze(0).cpu().numpy()
        pi3_p = m3(Xf_t, seed_t, forecast_steps=n).squeeze(0).cpu().numpy()
        b3_p  = b3_(Xf_t, seed_t, forecast_steps=n).squeeze(0).cpu().numpy()

    if return_predictions:
        return {
            'available_idx': avail.index.values,
            'forecast_idx':  fw.index.values,
            'true_values':   fw[target].values,
            'S1': {'pi': pi1_p, 'baseline': b1_p},
            'S2': {'pi': pi2_p, 'baseline': b2_p},
            'S3': {'pi': pi3_p, 'baseline': b3_p},
        }

    # ————————————————— Plotting (unchanged) —————————————————
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100, sharey=True)
    scenario_info = [
        ("S1: Fixed-Horizon Forecasting",   pi1_p, b1_p, 'd', 's'),
        ("S2: Recursive Forecasting",       pi2_p, b2_p, 'd', 's'),
        ("S3: Maximum-Horizon Forecasting", pi3_p, b3_p, 'd', 's')
    ]

    last_idx, last_val = avail.index[-1], avail[target].iloc[-1]
    first_idx, first_val = fw.index[0], fw[target].iloc[0]

    for ax, (title, pi_pred, base_pred, pi_m, base_m) in zip(axes, scenario_info):
        ax.axvline(x=forecast_rpt-1, color='black', linestyle='--', linewidth=1)
        ax.plot(avail.index, avail[target], 'ko-', markersize=9, linewidth=1.5, label='Data Available')
        ax.plot([last_idx, first_idx], [last_val, first_val], 'k-', linewidth=1.5)
        ax.plot(fut.index, fut[target], 'ko-', markersize=9, linewidth=1.5,
                markerfacecolor='white', label='True Capacity')
        ax.plot(fw.index, pi_pred, marker=pi_m, color='crimson',
                markersize=6, linestyle='-', linewidth=0.5, label='PI-RNN')
        ax.plot(fw.index, base_pred, marker=base_m, color='crimson',
                markersize=6, linestyle='--', linewidth=0.5, label='Baseline RNN')
        ax.set_xlabel('RPT Number (-)', fontsize=18)
        ax.set_xticks(np.arange(0, 35, 5))
        ax.set_title(title, fontsize=16)
        ax.legend(loc='upper right', fontsize=12)

    axes[0].set_ylabel('Capacity (Ah)', fontsize=18)
    axes[0].set_yticks(np.arange(0.4, 1.6, 0.2))
    axes[0].set_xlim(-2, 35)
    axes[0].set_ylim(0.4, 1.4)

    plt.tight_layout()
    plt.show()

# —————————————————————————————
# If run as a script, launch viz
# —————————————————————————————
if __name__ == '__main__':
    visualize_all_scenarios(
        forecast_rpt      = 10,
        forecast_steps    = 7,
        scaler            = scaler,
        features          = features,
        target            = target,
        Group             = 'G3',
        Cell              = 'C1',
        return_predictions= True,
        fine_tune         = False
    )
