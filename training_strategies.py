# training_strategies.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from data_utils import (
    prepare_pbm_surrogate,
    prepare_battery_sequences,
    create_sequences
)
from models import (
    MultiStepPIRNN,
    BaselineMultiStepRNN
)

# —————————————————————————————
# 0. Styling & Reproducibility
# —————————————————————————————
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 18

seed = 40
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# —————————————————————————————
# 1. Train PBM surrogate for PI-RNN injection
# —————————————————————————————
file_paths = [
    'Simulated_PBM_data/G18_PBM_Simulated.pkl',
    'Simulated_PBM_data/G16_PBM_Simulated.pkl',
    'Simulated_PBM_data/G4_PBM_Simulated.pkl',
    'Simulated_PBM_data/G3_PBM_Simulated.pkl',
    'Simulated_PBM_data/G2_PBM_Simulated.pkl'
]
sim_features_full = [
    'Ampere-Hour Throughput (Ah)',
    'Total Time Elapsed (h)',
    'Total Absolute Time From Start (h)',
    'Time Below 3A (h)',
    'Time Between 3A and 4A (h)',
    'Time Above 4A (h)',
    'RPT Number',
    'Capacity'
]
sim_target = "Capacity_Drop_Ah"

rf_model, scaler_sim = prepare_pbm_surrogate(
    file_paths,
    sim_features_full,
    sim_target,
    seed=seed
)

# —————————————————————————————
# 2. Prepare Battery Data & Create Sequences
# —————————————————————————————
features = [
    'Ampere-Hour Throughput (Ah)',
    'Total Absolute Time From Start (h)',
    'Total Time Elapsed (h)',
    'Time Below 3A (h)',
    'Time Between 3A and 4A (h)',
    'Time Above 4A (h)',
    'RPT Number'
]
target = 'Capacity'
batch1_path = 'Processed_data/Processed_data_Cycling&RPT_Batch1_Capacity_Forecasting_merged_update_Jan2025.pkl'
batch2_path = 'Processed_data/Processed_data_Cycling&RPT_Batch2_Capacity_Forecasting_merged_update_Jan2025.pkl'

X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df = \
    prepare_battery_sequences(
        batch1_path, batch2_path,
        features, target,
        val_cell_count=3,
        seed=seed
    )

# helper to build sequences
def make_seq(X_s, y, h):
    return create_sequences(X_s, y, h)

# horizons for scenarios
h1, h2, h3 = 7, 1, 10
X_tr1, y_tr1 = make_seq(X_train_s, y_train, h1)
X_va1, y_va1 = make_seq(X_val_s,   y_val,   h1)
X_tr2, y_tr2 = make_seq(X_train_s, y_train, h2)
X_va2, y_va2 = make_seq(X_val_s,   y_val,   h2)
X_tr3, y_tr3 = make_seq(X_train_s, y_train, h3)
X_va3, y_va3 = make_seq(X_val_s,   y_val,   h3)

# to tensors
def t(x): return torch.tensor(x, dtype=torch.float32)
X_tr1_t, y_tr1_t = t(X_tr1), t(y_tr1)
X_va1_t, y_va1_t = t(X_va1), t(y_va1)
X_tr2_t, y_tr2_t = t(X_tr2), t(y_tr2)
X_va2_t, y_va2_t = t(X_va2), t(y_va2)
X_tr3_t, y_tr3_t = t(X_tr3), t(y_tr3)
X_va3_t, y_va3_t = t(X_va3), t(y_va3)

# —————————————————————————————
# 3. Training helper
# —————————————————————————————
def train_scenario(model, optimizer, Xtr, ytr, Xva, yva, h, name):
    best_val, no_imp = float('inf'), 0
    criterion = nn.MSELoss()
    for ep in range(1, 2501):
        model.train(); optimizer.zero_grad()
        p = model(Xtr, ytr[:, :1], forecast_steps=h)
        loss = criterion(p, ytr[:, :h])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = model(Xva, yva[:, :1], forecast_steps=h)
            vloss = criterion(vp, yva[:, :h])

        if vloss < best_val:
            best_val, no_imp = vloss, 0
        else:
            no_imp += 1
            if no_imp >= 150:
                print(f"[{name}] early stop @epoch {ep}")
                break

        if ep % 50 == 0:
            print(f"[{name}] Epoch {ep} train={loss:.4f} val={vloss:.4f}")

# —————————————————————————————
# 4. Scenario 1: max-horizon (h1)
# —————————————————————————————
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
input_size  = len(features)+1
hidden_size = 50
pi1  = MultiStepPIRNN(input_size, hidden_size, rf_model)
opt1 = optim.Adam(pi1.parameters(), lr=1e-3)
train_scenario(pi1, opt1, X_tr1_t, y_tr1_t, X_va1_t, y_va1_t, h1, "PI-RNN S1")

np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
b1  = BaselineMultiStepRNN(input_size, hidden_size)
ob1 = optim.Adam(b1.parameters(), lr=1e-3)
train_scenario(b1, ob1, X_tr1_t, y_tr1_t, X_va1_t, y_va1_t, h1, "Base-RNN S1")

# —————————————————————————————
# 5. Scenario 2: recursive single-step (h2)
# —————————————————————————————
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
pi2  = MultiStepPIRNN(input_size, hidden_size, rf_model)
opt2 = optim.Adam(pi2.parameters(), lr=1e-3)
train_scenario(pi2, opt2, X_tr2_t, y_tr2_t, X_va2_t, y_va2_t, h2, "PI-RNN S2")

np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
b2  = BaselineMultiStepRNN(input_size, hidden_size)
ob2 = optim.Adam(b2.parameters(), lr=1e-3)
train_scenario(b2, ob2, X_tr2_t, y_tr2_t, X_va2_t, y_va2_t, h2, "Base-RNN S2")

# —————————————————————————————
# 6. Scenario 3: longer-horizon (h3)
# —————————————————————————————
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
pi3  = MultiStepPIRNN(input_size, hidden_size, rf_model)
opt3 = optim.Adam(pi3.parameters(), lr=1e-3)
train_scenario(pi3, opt3, X_tr3_t, y_tr3_t, X_va3_t, y_va3_t, h3, "PI-RNN S3")

np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
b3  = BaselineMultiStepRNN(input_size, hidden_size)
ob3 = optim.Adam(b3.parameters(), lr=1e-3)
train_scenario(b3, ob3, X_tr3_t, y_tr3_t, X_va3_t, y_va3_t, h3, "Base-RNN S3")

# —————————————————————————————
# 7. Recursive forecast helper (for S2)
# —————————————————————————————
def recursive_forecast(model, Xf, seed_cap, steps):
    model.eval()
    h = torch.zeros(1, model.hidden_size)
    cap = seed_cap.clone()
    preds = []
    with torch.no_grad():
        for t in range(steps):
            inp = torch.cat((Xf[:, t, :], cap), dim=1)
            h, drop = model.rnn_cell(inp, h)
            cap = cap - drop
            preds.append(cap.squeeze(-1))
    return torch.stack(preds,1)

def visualize_all_scenarios(
    forecast_rpt, forecast_steps, scaler, features, target,
    Group='G3', Cell='C1'
):
    """
    Three-panel plot of PI-RNN vs Baseline RNN forecasting strategies,
    exactly matching your original formatting.
    """
    # 1) select & sort cell data
    df = test_df[(test_df['Group']==Group) & (test_df['Cell']==Cell)].copy()
    df.sort_values('RPT Number', inplace=True); df.reset_index(drop=True, inplace=True)

    # 2) insert RPT=23 if missing (interpolate)
    if 23 not in df['RPT Number'].values:
        if {22,24}.issubset(df['RPT Number'].values):
            v22 = df.loc[df['RPT Number']==22, target].item()
            v24 = df.loc[df['RPT Number']==24, target].item()
            extra = pd.DataFrame({'RPT Number':[23], target:[(v22+v24)/2]})
            df = pd.concat([df, extra], ignore_index=True)
            df.sort_values('RPT Number', inplace=True); df.reset_index(drop=True, inplace=True)

    # 3) split into available vs future
    available_df = df[df['RPT Number'] <= forecast_rpt]
    future_df    = df[df['RPT Number'] >  forecast_rpt]

    # 4) clip to forecast_steps
    forecast_window_df = future_df.iloc[:forecast_steps]
    n_steps = len(forecast_window_df)

    # 5) build tensor of future features
    Xf = scaler.transform(forecast_window_df[features].values)
    Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0)

    # 6) seed capacity
    seed_val = available_df[target].iloc[-1]
    seed_t   = torch.tensor([[seed_val]], dtype=torch.float32)

    # 7) run all three models
    with torch.no_grad():
        pi1_p = pi1 (Xf_t, seed_t, forecast_steps=n_steps).squeeze(0).cpu().numpy()
        b1_p  = b1  (Xf_t, seed_t, forecast_steps=n_steps).squeeze(0).cpu().numpy()
        pi2_p = recursive_forecast(pi2, Xf_t, seed_t, steps=n_steps).squeeze(0).cpu().numpy()
        b2_p  = recursive_forecast(b2,  Xf_t, seed_t, steps=n_steps).squeeze(0).cpu().numpy()
        pi3_p = pi3 (Xf_t, seed_t, forecast_steps=n_steps).squeeze(0).cpu().numpy()
        b3_p  = b3  (Xf_t, seed_t, forecast_steps=n_steps).squeeze(0).cpu().numpy()

    # 8) plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100, sharey=True, constrained_layout=True)
    scenario_info = [
        ("S1: Fixed-Horizon Forecasting",   pi1_p, b1_p, 'd', 's', 'crimson', 'crimson'),
        ("S2: Recursive Forecasting",       pi2_p, b2_p, 'd', 's', 'crimson', 'crimson'),
        ("S3: Maximum-Horizon Forecasting", pi3_p, b3_p, 'd', 's', 'crimson', 'crimson'),
    ]

    last_idx      = available_df.index[-1]
    last_val      = available_df[target].iloc[-1]
    first_idx     = forecast_window_df.index[0]
    first_val     = forecast_window_df[target].iloc[0]

    for ax, (title, pi_pred, base_pred, pi_m, base_m, pi_c, base_c) \
            in zip(axes, scenario_info):
        # vertical forecast line
        ax.axvline(x=forecast_rpt-1, color='black', linestyle='--', linewidth=1)

        # available data
        ax.plot(
            available_df.index, available_df[target],
            marker='o', color='black', markersize=9,
            linestyle='-', linewidth=1.5, label='Data Available'
        )
        # connector
        ax.plot(
            [last_idx, first_idx], [last_val, first_val],
            color='black', linestyle='-', linewidth=1.5
        )
        # true future
        ax.plot(
            future_df.index, future_df[target],
            marker='o', color='black', markersize=9,
            linestyle='-', linewidth=1.5,
            label='True Capacity', markerfacecolor='white'
        )
        # PI-RNN
        ax.plot(
            forecast_window_df.index, pi_pred,
            marker=pi_m, color=pi_c,
            markersize=6, linestyle='-', linewidth=0.5,
            label='PI-RNN'
        )
        # Baseline RNN
        ax.plot(
            forecast_window_df.index, base_pred,
            marker=base_m, color=base_c,
            markersize=6, linestyle='--', linewidth=0.5,
            label='Baseline RNN'
        )

        ax.set_xlabel('RPT Number (-)', fontsize=18)
        ax.set_xticks(np.arange(0, 35, 5))
        ax.set_title(title, fontsize=16)
        ax.legend(loc='upper right', fontsize=12)

    axes[0].set_ylabel('Capacity (Ah)', fontsize=18)
    axes[0].set_yticks(np.arange(0.4, 1.6, 0.2))
    axes[0].set_xlim(-2, 35)
    axes[0].set_ylim(0.4, 1.4)

    # plt.tight_layout()
    plt.show()

# run the visualization
if __name__=='__main__':
    visualize_all_scenarios(
        forecast_rpt=10,
        forecast_steps=7,
        scaler=scaler,
        features=features,
        target=target,
        Group='G3',
        Cell='C1'
    )
