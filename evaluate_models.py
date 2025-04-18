# evaluate_models.py

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_models import run_training  # This re-trains surrogate + RNNs

# 0) Plot style
plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['font.size']        = 20
plt.rcParams['axes.labelsize']   = 18
plt.rcParams['axes.titlesize']   = 24
plt.rcParams['xtick.labelsize']  = 18
plt.rcParams['ytick.labelsize']  = 18

# 1) Train models (surrogate + PI‑RNN + baseline RNN) and get test data
pi_model, baseline_model, X_test, y_test_seq, max_horizon = run_training()

# 2) Prepare tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

# 3) Helper for single‑ and multi‑step evaluation
def eval_horizon(model, X_t, y_t, horizon):
    """
    Returns true and predicted for first 'horizon' steps, flattened.
    """
    model.eval()
    with torch.no_grad():
        seed = y_t[:, 0].unsqueeze(1)
        preds = model(X_t, seed, horizon)  # [batch, horizon]
    true = y_t[:, :horizon].cpu().numpy().flatten()
    pred = preds.cpu().numpy().flatten()
    return true, pred

def rmse(true, pred):
    return np.sqrt(np.mean((true - pred)**2))

# 4) Single‑step evaluation
true1_pi,  pred1_pi   = eval_horizon(pi_model,   X_test_tensor, y_test_tensor, 1)
true1_base, pred1_base= eval_horizon(baseline_model, X_test_tensor, y_test_tensor, 1)

rmse1_pi   = rmse(true1_pi,   pred1_pi)
rmse1_base = rmse(true1_base, pred1_base)

print(f"Single‑step RMSE → PI‑RNN: {rmse1_pi:.4f}, Baseline‑RNN: {rmse1_base:.4f}")

# Plot single‑step scatter for PI‑RNN
plt.figure(figsize=(5,4), dpi=300)
plt.scatter(true1_pi, pred1_pi, s=85, c='green', marker='s', edgecolors='w', alpha=0.8)
plt.plot([true1_pi.min(), true1_pi.max()],
         [true1_pi.min(), true1_pi.max()],
         'k--', lw=2)
plt.xlabel("True Capacity"); plt.ylabel("Predicted Capacity")
plt.text(0.05,0.95,
         f"RMSE {rmse1_pi:.4f}",
         transform=plt.gca().transAxes, va='top',
         bbox=dict(facecolor='wheat', alpha=0.5))
plt.grid(True)

# Plot single‑step scatter for Baseline RNN
plt.figure(figsize=(5,4), dpi=300)
plt.scatter(true1_base, pred1_base, s=85, c='orange', marker='s', edgecolors='w', alpha=0.8)
plt.plot([true1_base.min(), true1_base.max()],
         [true1_base.min(), true1_base.max()],
         'k--', lw=2)
plt.xlabel("True Capacity"); plt.ylabel("Predicted Capacity")
plt.text(0.05,0.95,
         f"RMSE {rmse1_base:.4f}",
         transform=plt.gca().transAxes, va='top',
         bbox=dict(facecolor='wheat', alpha=0.5))
plt.grid(True)

# 5) Multi‑step evaluation: horizons 2…10
horizons = list(range(2, 11))
rmse_pi_list = []
rmse_base_list = []

with torch.no_grad():
    for h in horizons:
        t_pi, p_pi     = eval_horizon(pi_model,   X_test_tensor, y_test_tensor, h)
        t_base, p_base = eval_horizon(baseline_model, X_test_tensor, y_test_tensor, h)
        rmse_pi_list.append(  rmse(t_pi,   p_pi)   )
        rmse_base_list.append(rmse(t_base, p_base))

# 6) Plot grouped bar chart
ind   = np.arange(len(horizons))
width = 0.35

fig, ax = plt.subplots(figsize=(10,4), dpi=300)
ax.bar(ind - width/2, rmse_pi_list,   width, color='green',    label='PI‑RNN')
ax.bar(ind + width/2, rmse_base_list, width, color='orange', label='Baseline RNN')

ax.set_xlabel('Forecasting Horizon (steps)')
ax.set_ylabel('RMSE')
ax.set_xticks(ind)
ax.set_xticklabels(horizons)
ax.legend()

plt.tight_layout()
plt.show()
