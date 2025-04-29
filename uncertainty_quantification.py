"""
Perform uncertainty quantification for PI-RNN capacity forecasts using MC-dropout,
apply isotonic recalibration on a held-out calibration set, and visualize both
trajectory prediction intervals and calibration curves.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
import os 
from data_utils import (
    PBM_SIM_PATHS,
    PBM_FEATURES,
    PBM_TARGET,
    BATTERY_FEATURES,
    BATTERY_TARGET,
    load_battery_data,
    load_batch,
    make_sequences,
    BATCH2_PATH
)
from models import train_pbm_surrogate_for_PI_RNN, MultiStepPIRNN

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 20

# —————————————————————————————
# 0. Seeds for reproducibility
# —————————————————————————————
seed = 40
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# —————————————————————————————
# 1. Train PBM surrogate (capacity‐drop)
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
# 2b. Build calibration set from batch2 (“C2” cells)
# —————————————————————————————
calibration_df = load_batch(BATCH2_PATH)
calibration_df = (
    calibration_df[ calibration_df['Cell']=='C2' ]
    .dropna()
    .sort_values('RPT Number')
    .reset_index(drop=True)
)

# —————————————————————————————
# 3. Build sequences for Scenario-3 (10-step)
# —————————————————————————————
h3 = 10
X_tr3, y_tr3 = make_sequences(X_train_s, y_train, h3)
X_va3, y_va3 = make_sequences(X_val_s,   y_val,   h3)

T = lambda x: torch.tensor(x, dtype=torch.float32)
X_tr3_t, y_tr3_t = T(X_tr3), T(y_tr3)
X_va3_t, y_va3_t = T(X_va3), T(y_va3)

# —————————————————————————————
# 4. Instantiate & train PI-RNN (max-horizon)
# —————————————————————————————
input_size  = len(features) + 1
hidden_size = 50

os.makedirs('saved_models', exist_ok=True)
model_path = 'saved_models/pi_rnn_s3_mc_dropout.pt'

model = MultiStepPIRNN(input_size, hidden_size, rf_model, dropout_rate=0.1)

if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded pretrained PI-RNN from {model_path}")
else:
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    mse   = nn.MSELoss()
    best_val, no_imp = float('inf'), 0

    for ep in range(1, 2501):
        model.train(); opt.zero_grad()
        pred  = model(X_tr3_t, y_tr3_t[:, :1], forecast_steps=h3)
        loss  = mse(pred, y_tr3_t)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vpred = model(X_va3_t, y_va3_t[:, :1], forecast_steps=h3)
            vloss = mse(vpred, y_va3_t)

        if vloss < best_val:
            best_val, no_imp = vloss, 0
        else:
            no_imp += 1
            if no_imp >= 150:
                print(f"[PI-RNN S3] early stop @ epoch {ep}")
                break
        if ep % 100 == 0:
            print(f"[PI-RNN S3] {ep}, train={loss:.4f}, val={vloss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Trained PI-RNN saved to {model_path}")

# —————————————————————————————
# 5. MC-dropout inference
# —————————————————————————————
def mc_dropout_forecast(model, Xf, seed_cap, steps, samples=20):
    model.train()
    sims = []
    with torch.no_grad():
        for _ in range(samples):
            sims.append(model(Xf, seed_cap, steps).cpu().numpy())
    sims = np.stack(sims, axis=0)   # (samples, batch, steps)
    mean = sims.mean(axis=0)        # (batch, steps)
    lb   = np.percentile(sims,  2.5, axis=0)
    ub   = np.percentile(sims, 97.5, axis=0)
    return mean, lb, ub

# —————————————————————————————
# 6. Trajectory helper & panel plot
# —————————————————————————————
def get_trajectory_predictions(model, scaler, feats, targ, df, rpt, hor, samples):
    ava = df[df['RPT Number'] <= rpt]
    fut = df[df['RPT Number'] >  rpt].iloc[:hor]
    Xf  = scaler.transform(fut[feats].fillna(0).values)
    Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0)
    seed = torch.tensor([[ava[targ].iloc[-1]]], dtype=torch.float32)
    m, l, u = mc_dropout_forecast(model, Xf_t, seed, len(fut), samples)
    return ava, fut, fut.index.to_numpy(), m.squeeze(0), l.squeeze(0), u.squeeze(0)

def plot_panel(ax, idx, ava, fut, targ, mean, lb, ub, rpt):
    ax.plot(ava.index, ava[targ],
            'ko-', mfc='black', ms=8, lw=1.5, label='Data Available')
    ax.plot([ava.index[-1], idx[0]],
            [ava[targ].iloc[-1], fut[targ].iloc[0]],
            'k-', lw=1.5)
    ax.plot(fut.index, fut[targ],
            'ko-', mfc='white', ms=8, lw=1.5, label='True Capacity')
    ax.plot(idx, mean, marker='d', color='crimson', linestyle='-', label='Prediction Interval', linewidth=0.5, markersize=4)
    errs = (ub - lb) / 2.0
    ax.errorbar(idx, mean, yerr=errs, fmt='none',
                ecolor='crimson', capsize=3, linewidth=0.75)
    ax.axvline(x=rpt-1, color='k', linestyle='--', linewidth=1)

# —————————————————————————————
# 7. Fit isotonic recalibration on C2 cells
# —————————————————————————————
def fit_isotonic_recalibration(model, calib_df, feats, targ, hor, samples):
    Xc, yc = make_sequences(calib_df[feats].values, calib_df[targ].values, hor)
    Xt, yt = torch.tensor(Xc, dtype=torch.float32), torch.tensor(yc, dtype=torch.float32)
    m, l, u = mc_dropout_forecast(model, Xt, yt[:,:1], hor, samples)
    final_mean = m[:, -1].flatten()
    true_final = yt[:, -1].numpy().flatten()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(final_mean, true_final)
    return iso

# —————————————————————————————
# 8. Plot convex-combined trajectories (three panels)
# —————————————————————————————
def plot_convex_combined_trajectories(
    model, scaler, feats, targ, df,
    rpts, hors, hor_train, samples, iso_reg,
    alpha=0.15, std_scaling=0.45
):
    phases = ["(First-Life)", "(Transition Phase)", "(Second-Life)"]
    fig, axes = plt.subplots(1, 3, figsize=(18,5), dpi=100)
    for i, (rpt, hor) in enumerate(zip(rpts, hors)):
        ava, fut, idx, m, l, u = get_trajectory_predictions(
            model, scaler, feats, targ, df, rpt, hor, samples
        )
        std0 = (u - l) / 2.0
        rec  = iso_reg.predict(m)
        comb = (1-alpha)*m + alpha*rec
        stdc = np.maximum.accumulate(std0) * std_scaling

        ax = axes[i]
        plot_panel(ax, idx, ava, fut, targ,
                   comb, comb-stdc, comb+stdc, rpt)
        ax.set_title(f"{df['Group'].iloc[0]}{df['Cell'].iloc[0]} {phases[i]}", fontsize=18)
        ax.set_xlabel('RPT Number (-)')
        if i==0:
            ax.set_ylabel('Capacity (Ah)')
        ax.set_xlim(-2,35);  ax.set_ylim(0.4,1.4)
        ax.set_xticks(np.arange(0,35,5))
        ax.set_yticks(np.arange(0.4,1.6,0.2))
        ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# —————————————————————————————
# Calibration Curve Helpers
# —————————————————————————————
def mc_dropout_predictions(model, X_tensor, current_tensor, num_samples, forecast_steps):
    model.train()
    sims = []
    with torch.no_grad():
        for _ in range(num_samples):
            sims.append(model(X_tensor, current_tensor, forecast_steps=forecast_steps).cpu().numpy())
    sims = np.stack(sims, axis=0)
    return sims.mean(axis=0), sims.std(axis=0)

def extract_final_step(predictions):
    return predictions[:, -1]

def compute_calibration_curve(true_vals, pred_mean, pred_std, bins=10):
    conf_levels = np.linspace(0,1,bins)
    observed    = []
    for c in conf_levels:
        z     = norm.ppf((1+c)/2)
        lower = pred_mean - z*pred_std
        upper = pred_mean + z*pred_std
        observed.append(np.mean((true_vals>=lower)&(true_vals<=upper)))
    return conf_levels, np.array(observed)

def evaluate_and_recalibrate_multiple_forecast_steps(
    model, calibration_df, features, target,
    num_samples_list, forecast_steps_list, std_scaling=0.5
):
    def create_seqs(data, targ, steps):
        X,y = [],[]
        for i in range(len(data)-steps+1):
            X.append(data[i:i+steps])
            y.append(targ[i:i+steps])
        return np.array(X), np.array(y)

    recal_results = {}
    test_results  = {}

    for steps in forecast_steps_list:
        Xc, yc = create_seqs(
            calibration_df[features].values,
            calibration_df[target].values,
            steps
        )
        Xt = torch.tensor(Xc, dtype=torch.float32)
        yt = torch.tensor(yc, dtype=torch.float32)

        recal_results[steps] = {}
        test_results [steps] = {}

        for ns in num_samples_list:
            mean_c, std_c = mc_dropout_predictions(model, Xt, yt[:,:1], ns, steps)
            mf = extract_final_step(mean_c).flatten()
            sf = extract_final_step(std_c).flatten()
            tf = yt[:,-1].numpy().flatten()

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(mf, tf)

            recal_mean = iso.predict(mf)
            recal_std  = sf * std_scaling

            recal_results[steps][ns] = (tf, recal_mean, recal_std)
            test_results [steps][ns] = (tf, recal_mean, recal_std)

    return recal_results, test_results

def plot_calibration_curves(test_results, forecast_steps_list, num_samples_list, bins=10):
    base_green = "#228B22"
    alphas     = {5:1.0, 10:0.8, 20:0.6, 40:0.4}

    fig, axes = plt.subplots(2, 3, figsize=(11,6), dpi=100)
    axes = axes.flat

    for ax, step in zip(axes, forecast_steps_list):
        ece_vals = []
        for ns in num_samples_list:
            tf, rm, rs = test_results[step][ns]
            L = min(len(tf), len(rm), len(rs))
            tf, rm, rs = tf[:L], rm[:L], rs[:L]
            conf, obs = compute_calibration_curve(tf, rm, rs, bins=bins)
            ece_vals.append(np.mean(np.abs(obs - conf)))
            ax.plot(conf, obs, 'o-', color=base_green,
                    alpha=alphas[ns], markersize=7, linewidth=1.5,
                    label=f'{ns} samples')
        ax.plot([0,1],[0,1],'--', color='gray', linewidth=2)
        avg_ece = np.mean(ece_vals)
        ax.set_title(f'{step}-step (Avg ECE: {avg_ece:.3f})', fontsize=12)
        ax.set_xlabel('Expected Confidence', fontsize=12)
        ax.set_ylabel('Observed Confidence', fontsize=12)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    plt.show()

# —————————————————————————————
# 9. Main: calibrate & plot both UQ trajectories & calibration curves
# —————————————————————————————
if __name__ == '__main__':
    # Fit isotonic regressor
    print("Fitting isotonic recalibration...")
    iso_reg = fit_isotonic_recalibration(
        model, calibration_df, features, target,
        hor=h3, samples=40
    )

    # Prepare the cell to visualize
    vis = test_df[
        (test_df['Group']=='G13') &
        (test_df['Cell' ]=='C1')
    ].reset_index(drop=True)

    # (A) Plot convex-combined trajectories
    print("Plotting convex-combined trajectories...")
    plot_convex_combined_trajectories(
        model, scaler, features, target, vis,
        rpts=[1,8,18],
        hors=[7,10,14],
        hor_train=h3,
        samples=40,
        iso_reg=iso_reg,
        alpha=0.15,
        std_scaling=0.45
    )

    # (B) Plot calibration curves
    forecast_steps_list = [1,2,4,6,8,10]
    num_samples_list   = [5,10,20,40]
    print("Evaluating & plotting calibration curves...")
    _, test_results = evaluate_and_recalibrate_multiple_forecast_steps(
        model, calibration_df, features, target,
        num_samples_list=num_samples_list,
        forecast_steps_list=forecast_steps_list,
        std_scaling=0.45
    )
    plot_calibration_curves(
        test_results,
        forecast_steps_list=forecast_steps_list,
        num_samples_list=num_samples_list,
        bins=10
    )
