# trajectory_forecast.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import warnings
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
from models import train_pbm_surrogate_for_PI_RNN, MultiStepPIRNN, BaselineMultiStepRNN, GPRBaseline

warnings.filterwarnings('ignore')

# 0. Styling & reproducibility
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 20

seed = 40
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# 1. Train PBM surrogate
rf_model, scaler_sim = train_pbm_surrogate_for_PI_RNN(
    PBM_SIM_PATHS,
    PBM_FEATURES,
    PBM_TARGET,
    seed=seed
)

# 2. Load & preprocess battery data
X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df = \
    load_battery_data(seed=seed)

# Scenario‐3 constants
h3 = 10
input_size  = len(BATTERY_FEATURES) + 1
hidden_size = 50

# 3. Load pretrained Scenario‐3 models
pi3 = MultiStepPIRNN(input_size, hidden_size, rf_model)
pi3.load_state_dict(torch.load('saved_models/pi3_scenario3.pth'))
pi3.eval()

b3 = BaselineMultiStepRNN(input_size, hidden_size)
b3.load_state_dict(torch.load('saved_models/b3_scenario3.pth'))
b3.eval()

def trajectory_forecast(
    group: str,
    cell:  str,
    fine_tune: bool = False,
    fine_tune_epochs: int = 20,
    return_predictions: bool = False
):
    # select cell data
    df = (
        test_df
        .loc[lambda d: (d['Group']==group) & (d['Cell']==cell)]
        .sort_values('RPT Number')
        .reset_index(drop=True)
    )
    df[BATTERY_FEATURES] = df[BATTERY_FEATURES].fillna(0)

    # insert missing RPT=23 if needed
    if 23 not in df['RPT Number'].values and {22,24}.issubset(df['RPT Number'].values):
        v22 = df.loc[df['RPT Number']==22, BATTERY_TARGET].item()
        v24 = df.loc[df['RPT Number']==24, BATTERY_TARGET].item()
        extra = pd.DataFrame({'RPT Number':[23], BATTERY_TARGET:[(v22+v24)/2]})
        df = pd.concat([df, extra], ignore_index=True) \
               .sort_values('RPT Number') \
               .reset_index(drop=True)

    # available and future splits for scenario 3 forecasting
    avail = df[df['RPT Number'] <= h3]
    fut   = df[df['RPT Number'] >  h3]
    fw    = fut.iloc[:h3]           # forecast window size = h3
    n     = len(fw)

    # handle fine-tuning on first 5 points of this cell
    if fine_tune and len(avail) >= 10:
        # clone models
        m_clone = deepcopy(pi3)
        b_clone = deepcopy(b3)
        m_clone.train()
        b_clone.train()

        # prepare fine-tune data: first 5 points
        raw_feats = avail[BATTERY_FEATURES].values[:10]
        raw_tgts  = avail[BATTERY_TARGET].values[:10]
        scaled_5  = scaler.transform(raw_feats)

        # make a 5-step sequence for fine-tuning
        Xm, ym = make_sequences(scaled_5, raw_tgts, 10)
        Xm_t = torch.tensor(Xm, dtype=torch.float32)
        ym_t = torch.tensor(ym, dtype=torch.float32)

        opt_m = optim.Adam(m_clone.parameters(), lr=1e-3)
        opt_b = optim.Adam(b_clone.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for _ in range(fine_tune_epochs):
            # fine-tune PI-RNN S3
            opt_m.zero_grad()
            p3 = m_clone(Xm_t, ym_t[:, :1], forecast_steps=10)
            l3 = loss_fn(p3, ym_t)
            l3.backward()
            opt_m.step()

            # fine-tune Baseline S3
            opt_b.zero_grad()
            bp = b_clone(Xm_t, ym_t[:, :1], forecast_steps=10)
            lb = loss_fn(bp, ym_t)
            lb.backward()
            opt_b.step()

        m_clone.eval()
        b_clone.eval()
        model = m_clone
        baseline = b_clone
    else:
        model = pi3
        baseline = b3

    # 4. Precompute GPR trajectory
    gpr = GPRBaseline(initial_points=10)
    gpr.fit(df, (group, cell), initial_points=10)

    hybrid = []
    for origin in range(len(df)):
        if origin + 1 < len(df):
            _, yp = gpr.predict(
                df.iloc[origin:].reset_index(drop=True),
                (group, cell),
                initial_points=1
            )
            hybrid.append(yp[0])
        else:
            hybrid.append(np.nan)
    y_pred_gpr = np.array(hybrid)

    # prepare future-window tensor for RNNs
    Xf = scaler.transform(fw[BATTERY_FEATURES].values)
    Xt = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0)
    St = torch.tensor([[avail[BATTERY_TARGET].iloc[-1]]], dtype=torch.float32)

    with torch.no_grad():
        bb = baseline(Xt, St, forecast_steps=n).cpu().numpy().squeeze(0)
        pp = model   (Xt, St, forecast_steps=n).cpu().numpy().squeeze(0)

    # return raw arrays if requested
    if return_predictions:
        return {
            'available_idx': avail.index.values,
            'forecast_idx':  fw.index.values,
            'true_values':   fw[BATTERY_TARGET].values,
            'gpr':            y_pred_gpr[h3:h3+n],
            'baseline_rnn':   bb,
            'pi_rnn':         pp
        }

    # 5. Final Plotting 
    forecast_rpts     = [1, 9, 23]
    forecast_horizons = [7, 13, 10]
    fixed_horizon     = 5
    red_contrast      = '#D62728'
    light_red         = '#F5B7B1'
    hatches           = ['///','\\\\','...']
    phases            = ["(First-Life)","(Transition Phase)","(Second-Life)"]

    fig = plt.figure(figsize=(16, 4), dpi=100, constrained_layout=True)
    gs  = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
    ax_top  = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_rmse = [fig.add_subplot(gs[1, i]) for i in range(3)]
    bar_width = 0.25

    # --- TOP PANELS ---
    for i, (rpt, horizon, phase) in enumerate(zip(forecast_rpts, forecast_horizons, phases)):
        ava  = df[df['RPT Number'] <= rpt]
        fut_i= df[df['RPT Number'] >  rpt]
        fw_i = fut_i.iloc[:horizon]
        n_i  = len(fw_i)

        ax = ax_top[i]
        ax.axvline(x=rpt-1, color='black', linestyle='--', linewidth=1)

        # available
        ax.plot(ava.index, ava[BATTERY_TARGET],
                'ko-', markersize=12, linewidth=1.5, label='Data Available')
        # connector
        if not ava.empty and not fut_i.empty:
            idx0, idx1 = ava.index[-1], fut_i.index[0]
            v0, v1     = ava[BATTERY_TARGET].iloc[-1], fut_i[BATTERY_TARGET].iloc[0]
            ax.plot([idx0, idx1], [v0, v1], 'k-', linewidth=1)

        # true future
        ax.plot(fut_i.index, fut_i[BATTERY_TARGET],
                'ko-', markersize=12, linewidth=1.5,
                markerfacecolor='white', label='True Capacity')

        # GPR
        gpr_fc = y_pred_gpr[rpt:rpt+n_i]
        ax.plot(fw_i.index, gpr_fc,
                marker='^', color='crimson', markersize=8,
                linestyle='-.', linewidth=0.5, label='Baseline GPR')

        # RNNs
        raw = fw_i[BATTERY_FEATURES].values
        Xf_i  = scaler.transform(raw)
        Xt_i  = torch.tensor(Xf_i, dtype=torch.float32).unsqueeze(0)
        St_i  = torch.tensor([[ava[BATTERY_TARGET].iloc[-1]]], dtype=torch.float32)
        with torch.no_grad():
            bb_i = baseline(Xt_i, St_i, forecast_steps=n_i).cpu().numpy().squeeze(0)
            pp_i = model   (Xt_i, St_i, forecast_steps=n_i).cpu().numpy().squeeze(0)

        ax.plot(fw_i.index, bb_i,
                's', color='crimson', markersize=8,
                linestyle='--', linewidth=0.5, label='Baseline RNN')
        ax.plot(fw_i.index, pp_i,
                'd', color='crimson', markersize=8,
                linestyle='-', linewidth=0.5, label='PI-RNN')

        ax.set_title(f"{group}{cell} {phase}", fontsize=24)
        ax.set_xlabel('RPT Number (-)', fontsize=20)
        ax.set_xticks(np.arange(0, 35, 5))
        ax.set_ylabel('Capacity (Ah)', fontsize=20)
        ax.set_yticks(np.arange(0.4, 1.6, 0.2))
        ax.set_xlim(-2, 35)
        ax.set_ylim(0.4, 1.4)
        ax.tick_params(labelsize=20)
        ax.legend(loc='upper right', fontsize=14)

    # --- BOTTOM PANELS: RMSE bars ---
    origins = df['RPT Number'].astype(int).values
    rmse_b, rmse_p, rmse_g = {}, {}, {}
    for origin in origins:
        ava = df[df['RPT Number'] <= origin]
        fut_i = df[df['RPT Number'] > origin].iloc[:fixed_horizon]
        if fut_i.empty: continue

        raw = fut_i[BATTERY_FEATURES].values
        Xf_i = scaler.transform(raw)
        Xt_i = torch.tensor(Xf_i, dtype=torch.float32).unsqueeze(0)
        St_i = torch.tensor([[ava[BATTERY_TARGET].iloc[-1]]], dtype=torch.float32)
        with torch.no_grad():
            bpr = baseline(Xt_i, St_i, forecast_steps=len(fut_i)).cpu().numpy().squeeze(0)
            ppr = model   (Xt_i, St_i, forecast_steps=len(fut_i)).cpu().numpy().squeeze(0)
        gslice = y_pred_gpr[origin:origin+len(fut_i)]
        true_vals = fut_i[BATTERY_TARGET].values

        rmse_b[origin] = np.sqrt(((bpr-true_vals)**2).mean())
        rmse_p[origin] = np.sqrt(((ppr-true_vals)**2).mean())
        rmse_g[origin] = np.sqrt(((gslice-true_vals)**2).mean())

    complete_rpts      = np.arange(0, origins.max()+1)
    full_rmse_b = np.array([rmse_b.get(r, np.nan) for r in complete_rpts])
    full_rmse_p = np.array([rmse_p.get(r, np.nan) for r in complete_rpts])
    full_rmse_g = np.array([rmse_g.get(r, np.nan) for r in complete_rpts])
    for arr in (full_rmse_b, full_rmse_p, full_rmse_g):
        if np.isnan(arr[0]) and len(arr)>1:
            arr[0] = arr[1]

    for ax in ax_rmse:
        x = complete_rpts
        ax.bar(x - bar_width, full_rmse_g, bar_width,
               edgecolor=red_contrast, color=light_red, hatch=hatches[2],
               label='Baseline GPR')
        ax.bar(x,           full_rmse_b, bar_width,
               edgecolor=red_contrast, color=light_red, hatch=hatches[0],
               label='Baseline RNN')
        ax.bar(x + bar_width, full_rmse_p, bar_width,
               edgecolor=red_contrast, color=light_red, hatch=hatches[1],
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='G13')
    parser.add_argument('--cell',  type=str, default='C1')
    parser.add_argument('--fine-tune', action='store_true')
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--return-predictions', action='store_true')
    args = parser.parse_args()

    out = trajectory_forecast(
        args.group,
        args.cell,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.epochs,
        return_predictions=args.return_predictions
    )
    if args.return_predictions:
        print(out)
