# trajectory_forecast.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import warnings

from models import MultiStepPIRNN, BaselineMultiStepRNN, GPRBaseline
from data_utils import prepare_battery_sequences, prepare_pbm_surrogate

warnings.filterwarnings('ignore')

# —————————————————————————————
# 0. Styling & reproducibility
# —————————————————————————————
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 20

seed = 40
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# —————————————————————————————
# 1. Train PBM surrogate
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
# 2. Load & preprocess battery data
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
batch1 = 'Processed_data/Processed_data_Cycling&RPT_Batch1_Capacity_Forecasting_merged_update_Jan2025.pkl'
batch2 = 'Processed_data/Processed_data_Cycling&RPT_Batch2_Capacity_Forecasting_merged_update_Jan2025.pkl'

X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df = \
    prepare_battery_sequences(
        batch1, batch2,
        features, target,
        val_cell_count=3,
        seed=seed
    )

# —————————————————————————————
# 3. Scenario 3 training (10-step forecasting)
# —————————————————————————————
h3 = 10
def create_seq(data, vals, h):
    X, Y = [], []
    for i in range(len(data) - h + 1):
        X.append(data[i : i + h])
        Y.append(vals[i : i + h])
    return np.array(X), np.array(Y)

X_tr3, y_tr3 = create_seq(X_train_s, y_train, h3)
X_va3, y_va3 = create_seq(X_val_s,   y_val,   h3)

X_tr3_t = torch.tensor(X_tr3, dtype=torch.float32)
y_tr3_t = torch.tensor(y_tr3, dtype=torch.float32)
X_va3_t = torch.tensor(X_va3, dtype=torch.float32)
y_va3_t = torch.tensor(y_va3, dtype=torch.float32)

input_size  = len(features) + 1
hidden_size = 50

pi3  = MultiStepPIRNN(input_size, hidden_size, rf_model)
b3   = BaselineMultiStepRNN(input_size, hidden_size)
opt3 = torch.optim.Adam(pi3.parameters(), lr=1e-3)
ob3  = torch.optim.Adam(b3.parameters(),  lr=1e-3)
criterion = nn.MSELoss()

# Train PI-RNN S3
best_val3, no_imp3 = float('inf'), 0
for ep in range(1, 2501):
    pi3.train(); opt3.zero_grad()
    preds = pi3(X_tr3_t, y_tr3_t[:, :1], forecast_steps=h3)
    loss  = criterion(preds, y_tr3_t[:, :h3])
    loss.backward(); opt3.step()
    pi3.eval()
    with torch.no_grad():
        vpred = pi3(X_va3_t, y_va3_t[:, :1], forecast_steps=h3)
        vloss = criterion(vpred, y_va3_t[:, :h3])
    if vloss < best_val3:
        best_val3, no_imp3 = vloss, 0
    else:
        no_imp3 += 1
        if no_imp3 >= 150:
            print(f"[PI-RNN S3] early stop @ epoch {ep}")
            break
    if ep % 100 == 0:
        print(f"[PI-RNN S3] {ep}, train={loss:.4f}, val={vloss:.4f}")

# Train Baseline RNN S3
best_vb3, no_impb3 = float('inf'), 0
for ep in range(1, 2501):
    b3.train(); ob3.zero_grad()
    bp = b3(X_tr3_t, y_tr3_t[:, :1], forecast_steps=h3)
    bl = criterion(bp, y_tr3_t[:, :h3])
    bl.backward(); ob3.step()
    b3.eval()
    with torch.no_grad():
        vb = b3(X_va3_t, y_va3_t[:, :1], forecast_steps=h3)
        vbl = criterion(vb, y_va3_t[:, :h3])
    if vbl < best_vb3:
        best_vb3, no_impb3 = vbl, 0
    else:
        no_impb3 += 1
        if no_impb3 >= 150:
            print(f"[Base-RNN S3] early stop @ epoch {ep}")
            break
    if ep % 100 == 0:
        print(f"[Base-RNN S3] {ep}, train={bl:.4f}, val={vbl:.4f}")

model_scenario3          = pi3
baseline_model_scenario3 = b3

# —————————————————————————————
# 4. Precompute GPR “hybrid” trajectory
# —————————————————————————————
group, cell = 'G8', 'C1'
cell_df = (
    test_df
    .loc[lambda d: (d['Group']==group)&(d['Cell']==cell)]
    .copy()
    .sort_values('RPT Number')
    .reset_index(drop=True)
)
cell_df[features] = cell_df[features].fillna(0)

# Insert RPT=23 if missing
if 23 not in cell_df['RPT Number'].values and {22,24}.issubset(cell_df['RPT Number'].values):
    v22 = cell_df.loc[cell_df['RPT Number']==22, target].item()
    v24 = cell_df.loc[cell_df['RPT Number']==24, target].item()
    extra = pd.DataFrame({'RPT Number':[23], target:[(v22+v24)/2]})
    cell_df = pd.concat([cell_df, extra], ignore_index=True)
    cell_df = cell_df.sort_values('RPT Number').reset_index(drop=True)

gpr = GPRBaseline(initial_points=10)
gpr.fit(cell_df, (group, cell), initial_points=10)

hybrid = []
for origin in range(len(cell_df)):
    if origin+1 < len(cell_df):
        _, yp = gpr.predict(
            cell_df.iloc[origin:].reset_index(drop=True),
            (group,cell),
            initial_points=1
        )
        hybrid.append(yp[0])
    else:
        hybrid.append(np.nan)
y_c1_pred_gpr = np.array(hybrid)

# —————————————————————————————
# 5. Final Plotting (exact original style)
# —————————————————————————————
forecast_rpts     = [1, 9, 23]
forecast_horizons = [7, 13, 10]
fixed_horizon     = 5
red_contrasting   = '#D62728'
light_red         = '#F5B7B1'
hatches           = ['///','\\\\','...']
phases            = ["(First-Life)","(Transition Phase)","(Second-Life)"]

fig = plt.figure(figsize=(20, 4), dpi=100, constrained_layout=True)
gs  = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
ax_top  = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_rmse = [fig.add_subplot(gs[1, i]) for i in range(3)]
bar_width = 0.25

# --- TOP PANELS ---
for i, (rpt, horizon, phase) in enumerate(zip(forecast_rpts, forecast_horizons, phases)):
    available_df       = cell_df[cell_df['RPT Number'] <= rpt]
    future_df          = cell_df[cell_df['RPT Number'] >  rpt]
    forecast_window_df = future_df.iloc[:horizon]

    ax = ax_top[i]

    # vertical line at origin
    ax.axvline(x=rpt-1, color='black', linestyle='--', linewidth=1)

    # available data
    ax.plot(available_df.index, available_df[target],
            marker='o', color='black', markersize=12,
            linestyle='-', linewidth=1.5, label='Data Available')

    # connecting line
    if not available_df.empty and not future_df.empty:
        last_idx = available_df.index[-1]
        first_idx = future_df.index[0]
        last_val = available_df[target].iloc[-1]
        first_val= future_df[target].iloc[0]
        ax.plot([last_idx, first_idx], [last_val, first_val],
                color='black', linestyle='-', linewidth=1)

    # true future
    ax.plot(future_df.index, future_df[target],
            marker='o', color='black', markersize=12,
            linestyle='-', linewidth=1.5,
            markerfacecolor='white', label='True Capacity')

    # GPR
    gpr_forecast = y_c1_pred_gpr[rpt : rpt + len(forecast_window_df)]
    ax.plot(forecast_window_df.index, gpr_forecast,
            marker='^', color='crimson', markersize=8,
            linestyle='-.', linewidth=0.5,
            label='Baseline GPR')

    # RNN & PI-RNN
    raw = forecast_window_df[features].fillna(0).values
    Xf  = scaler.transform(raw)
    Xt  = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0)
    St  = torch.tensor([[available_df[target].iloc[-1]]], dtype=torch.float32)
    with torch.no_grad():
        bb = baseline_model_scenario3(Xt, St, forecast_steps=len(forecast_window_df))\
                 .detach().cpu().numpy().squeeze(0)
        pp = model_scenario3         (Xt, St, forecast_steps=len(forecast_window_df))\
                 .detach().cpu().numpy().squeeze(0)

    ax.plot(forecast_window_df.index, bb,
            marker='s', color='crimson', markersize=8,
            linestyle='--', linewidth=0.5, label='Baseline RNN')
    ax.plot(forecast_window_df.index, pp,
            marker='d', color='crimson', markersize=8,
            linestyle='-', linewidth=0.5, label='PI-RNN')

    ax.set_title(f"{group}{cell} {phase}", fontsize=24)
    ax.set_xlabel('RPT Number (-)', fontsize=20)
    ax.set_xticks(np.arange(0, 35, 5))
    ax.set_ylabel('Capacity (Ah)', fontsize=20)
    ax.set_yticks(np.arange(0.4, 1.6, 0.2))
    ax.set_xlim(-2, 35)
    ax.set_ylim(0.4, 1.4)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(loc='upper right', fontsize=14)

# --- BOTTOM PANELS: RMSE bars ---
origins = cell_df['RPT Number'].astype(int).values
rmse_b, rmse_p, rmse_g = {}, {}, {}
for origin in origins:
    ava = cell_df[cell_df['RPT Number'] <= origin]
    fut = cell_df[cell_df['RPT Number'] > origin].iloc[:fixed_horizon]
    if fut.empty: continue

    raw = fut[features].fillna(0).values
    Xf  = scaler.transform(raw)
    Xt  = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0)
    St  = torch.tensor([[ava[target].iloc[-1]]], dtype=torch.float32)
    with torch.no_grad():
        bpr = baseline_model_scenario3(Xt, St, forecast_steps=len(fut))\
                  .detach().cpu().numpy().squeeze(0)
        ppr = model_scenario3         (Xt, St, forecast_steps=len(fut))\
                  .detach().cpu().numpy().squeeze(0)
    gpr_slice = y_c1_pred_gpr[origin : origin + len(fut)]
    true_vals = fut[target].values

    rmse_b[origin] = np.sqrt(np.mean((bpr       - true_vals)**2))
    rmse_p[origin] = np.sqrt(np.mean((ppr       - true_vals)**2))
    rmse_g[origin] = np.sqrt(np.mean((gpr_slice - true_vals)**2))

complete_rpts      = np.arange(0, max(origins)+1)
full_rmse_baseline = np.array([rmse_b.get(r, np.nan) for r in complete_rpts])
full_rmse_pi       = np.array([rmse_p.get(r, np.nan) for r in complete_rpts])
full_rmse_gpr      = np.array([rmse_g.get(r, np.nan) for r in complete_rpts])
for arr in (full_rmse_baseline, full_rmse_pi, full_rmse_gpr):
    if np.isnan(arr[0]) and len(arr)>1:
        arr[0] = arr[1]

for ax in ax_rmse:
    x = complete_rpts
    ax.bar(x - bar_width, full_rmse_gpr, bar_width,
           edgecolor=red_contrasting, color=light_red, hatch=hatches[2],
           label='Baseline GPR')
    ax.bar(x,          full_rmse_baseline, bar_width,
           edgecolor=red_contrasting, color=light_red, hatch=hatches[0],
           label='Baseline RNN')
    ax.bar(x + bar_width, full_rmse_pi, bar_width,
           edgecolor=red_contrasting, color=light_red, hatch=hatches[1],
           label='PI-RNN')

    ax.set_yscale('log')
    ax.set_xlabel('RPT Number (-)', fontsize=20)
    ax.set_ylabel('RMSE (Ah)', fontsize=20)
    ax.set_xticks(np.arange(0, 35, 5))
    ax.set_xlim(-2, 35)
    ax.legend(loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5,1.3))
    ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
