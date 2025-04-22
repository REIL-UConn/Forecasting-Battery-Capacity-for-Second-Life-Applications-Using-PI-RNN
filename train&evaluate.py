import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from models import train_pbm_surrogate, MultiStepPIRNN, BaselineMultiStepRNN

# —————————————————————————————
# 0. Styling & reproducibility
# —————————————————————————————
plt.rcParams['font.family']    = 'Times New Roman'
plt.rcParams['font.size']      = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize']= 18
plt.rcParams['ytick.labelsize']= 18

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

sim_features = [
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

rf_model, scaler_sim = train_pbm_surrogate(
    file_paths,
    sim_features,
    sim_target,
    seed=seed
)

#############################################
# 4. Prepare Battery Data & Create Sequences
#############################################

def load_batch(path):
    df = pd.read_pickle(path)
    df.sort_values(['Channel Number','Group','Cell','RPT Number'], inplace=True)
    df['capacity_drop'] = df.groupby(['Channel Number','Group','Cell'])['Capacity']\
                             .diff().abs().fillna(0)
    return df

# load both batches
batch1 = load_batch('Processed_data/Processed_data_Cycling&RPT_Batch1_Capacity_Forecasting_merged_update_Jan2025.pkl')
batch2 = load_batch('Processed_data/Processed_data_Cycling&RPT_Batch2_Capacity_Forecasting_merged_update_Jan2025.pkl')

# apply group filters
test_df  = batch1[~batch1['Group'].isin(['G12'])].dropna()
train_df = batch2[~batch2['Group'].isin(['G11','G14'])].dropna()

# only cells C1/C3 for train/val
train_df = train_df[train_df['Cell'].isin(['C1','C3'])]

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

# split train/val by unique cell ID
train_df['Unique_Cell_ID'] = train_df['Group'] + '-' + train_df['Cell']
ids = train_df['Unique_Cell_ID'].unique().tolist()
val_ids = random.sample(ids, 3)
validation_df = train_df[train_df['Unique_Cell_ID'].isin(val_ids)]
training_df   = train_df[~train_df['Unique_Cell_ID'].isin(val_ids)]

# extract and scale
scaler = MinMaxScaler().fit(training_df[features])
X_train_s = scaler.transform(training_df[features])
X_val_s   = scaler.transform(validation_df[features])
X_test_s  = scaler.transform(test_df[features])

y_train = training_df[target].values
y_val   = validation_df[target].values
y_test  = test_df[target].values

def create_sequences(arr, vals, steps):
    Xs, ys = [], []
    for i in range(len(arr) - steps + 1):
        Xs.append(arr[i:i+steps])
        ys.append(vals[i:i+steps])
    return np.array(Xs), np.array(ys)

forecast_steps = 10
X_train_seq, y_train_seq = create_sequences(X_train_s, y_train, forecast_steps)
X_val_seq,   y_val_seq   = create_sequences(X_val_s,   y_val,   forecast_steps)

# tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_seq,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_seq,   dtype=torch.float32)

#############################################
# 5. Train PI‑RNN & Baseline RNN
#############################################
# Both models use input_size = len(features) + 1 = 7 + 1 = 8.
input_size = len(features) + 1
hidden_size = 50
num_epochs = 2500
patience = 50

# PI‑RNN
# Initialize the PI-RNN model (max-horizon training).
model_PI_RNN = MultiStepPIRNN(input_size, hidden_size, rf_model)
optimizer_PI = optim.Adam(model_PI_RNN.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_PI = float('inf')
no_imp_PI   = 0
for epoch in range(1, num_epochs+1):
    model_PI_RNN.train(); optimizer_PI.zero_grad()
    seed = y_train_tensor[:, [0]]
    preds = model_PI_RNN(X_train_tensor, seed, forecast_steps)
    loss  = criterion(preds, y_train_tensor[:, :forecast_steps])
    loss.backward(); optimizer_PI.step()

    model_PI_RNN.eval()
    with torch.no_grad():
        val_seed = y_val_tensor[:, [0]]
        val_preds = model_PI_RNN(X_val_tensor, val_seed, forecast_steps)
        val_loss  = criterion(val_preds, y_val_tensor[:, :forecast_steps])

    if val_loss < best_val_PI:
        best_val_PI, no_imp_PI = val_loss, 0
    else:
        no_imp_PI += 1
        if no_imp_PI >= patience:
            print(f"Early stop PI-RNN at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"[PI-RNN] {epoch}/{num_epochs} - train {loss:.4f}, val {val_loss:.4f}")

# Baseline RNN
# Initialize the Baseline RNN model (max-horizon training).
baseline_model = BaselineMultiStepRNN(input_size, hidden_size)
optimizer_base = optim.Adam(baseline_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_base = float('inf')
no_imp_base  = 0
for epoch in range(1, num_epochs+1):
    baseline_model.train(); optimizer_base.zero_grad()
    seed = y_train_tensor[:, [0]]
    preds = baseline_model(X_train_tensor, seed, forecast_steps)
    loss  = criterion(preds, y_train_tensor[:, :forecast_steps])
    loss.backward(); optimizer_base.step()

    baseline_model.eval()
    with torch.no_grad():
        val_seed  = y_val_tensor[:, [0]]
        val_preds = baseline_model(X_val_tensor, val_seed, forecast_steps)
        val_loss  = criterion(val_preds, y_val_tensor[:, :forecast_steps])

    if val_loss < best_val_base:
        best_val_base, no_imp_base = val_loss, 0
    else:
        no_imp_base += 1
        if no_imp_base >= patience:
            print(f"Early stop Baseline RNN at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"[Baseline] {epoch}/{num_epochs} - train {loss:.4f}, val {val_loss:.4f}")


#############################################
# 6. Final Evaluation & Plotting (Single‑Step)
#############################################

def evaluate_single_step(model, X_seq, y_seq):
    model.eval()
    with torch.no_grad():
        seed = y_seq[:, [0]]
        preds_all = model(X_seq, seed, forecast_steps)
    return preds_all[:, 0].cpu().numpy()

X_test_seq, y_test_seq = create_sequences(X_test_s, y_test, forecast_steps)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
y_true = y_test_tensor[:, 0].cpu().numpy()

pi_preds   = evaluate_single_step(model_PI_RNN, X_test_tensor, y_test_tensor)
base_preds = evaluate_single_step(baseline_model, X_test_tensor, y_test_tensor)

def calc_metrics(y_t, y_p):
    return np.sqrt(((y_t - y_p)**2).mean()), np.mean(np.abs(y_t - y_p))

rmse_PI, mae_PI     = calc_metrics(y_true, pi_preds)
rmse_base, mae_base = calc_metrics(y_true, base_preds)

for preds, color, name, (rmse, mae) in [
    (pi_preds,   'green',   'PI-RNN',     (rmse_PI,   mae_PI)),
    (base_preds, 'orange', 'Baseline RNN',(rmse_base, mae_base))
]:
    plt.figure(figsize=(5, 4), dpi=300)
    plt.scatter(y_true, preds, s=85, c=color, marker='s', edgecolors='w', alpha=0.8)
    lims = [y_true.min(), y_true.max()]
    plt.plot(lims, lims, 'k--', lw=2)
    plt.xlabel('True Capacity')
    plt.ylabel('Predicted Capacity')
    plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
             transform=plt.gca().transAxes, va='top',
             bbox=dict(facecolor='wheat', alpha=0.5))
    plt.grid(True)

plt.show()

