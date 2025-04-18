import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# ──────────────────────────────────────────────────────────────────────────────
# 0) FIXED SEEDS & GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SEED = 40
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────────────
# 1) TRAIN RANDOM FOREST SURROGATE
# ──────────────────────────────────────────────────────────────────────────────
def train_surrogate():
    sim_paths = [
        '../physics-based model for data augmentation/CaseStudy2_Saved_Data/v2/par_reversible_plating_G18_revised_feature_set_FinalParameterSet.pkl',
        '../physics-based model for data augmentation/CaseStudy2_Saved_Data/v2/par_reversible_plating_G16_revised_feature_set_FinalParameterSet.pkl',
        '../physics-based model for data augmentation/CaseStudy2_Saved_Data/v2/par_reversible_plating_G4_revised_feature_set_FinalParameterSet.pkl',
        '../physics-based model for data augmentation/CaseStudy2_Saved_Data/v2/par_reversible_plating_G3_revised_feature_set_FinalParameterSet.pkl',
        '../physics-based model for data augmentation/CaseStudy2_Saved_Data/v2/par_reversible_plating_G2_revised_feature_set_FinalParameterSet.pkl',
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

    dfs = [pd.read_pickle(fp) for fp in sim_paths]
    df  = pd.concat(dfs, ignore_index=True)
    X   = df[sim_features].values
    y   = df[sim_target].values

    scaler = MinMaxScaler().fit(X)
    Xs     = scaler.transform(X)

    rf = RandomForestRegressor(n_estimators=200, random_state=SEED)
    rf.fit(Xs, y)
    return rf, scaler


# ──────────────────────────────────────────────────────────────────────────────
# 2) DEFINE PI‑RNN & BASELINE RNN
# ──────────────────────────────────────────────────────────────────────────────
class CustomRNNCellWithSurrogate(nn.Module):
    def __init__(self, input_size, hidden_size, surrogate):
        super().__init__()
        self.W_ih   = nn.Linear(input_size, hidden_size)
        self.W_hh   = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pbm_weight  = nn.Parameter(torch.randn(1, hidden_size))
        self.fc    = nn.Linear(hidden_size, 1)
        self.surrogate = surrogate

    def forward(self, x, h):
        feat = x.detach().cpu().numpy()
        pbm_out = self.surrogate.predict(feat)
        pbm_t   = torch.tensor(pbm_out, dtype=torch.float32, device=x.device)
        pbm_t   = pbm_t.unsqueeze(1).expand(-1, h.size(1))
        h_pbm   = self.pbm_weight * pbm_t

        h_next = self.activation(self.W_ih(x) + self.W_hh(h) + h_pbm)
        drop   = self.fc(h_next)
        return h_next, drop

class MultiStepPIRNN(nn.Module):
    def __init__(self, input_size, hidden_size, surrogate):
        super().__init__()
        self.cell = CustomRNNCellWithSurrogate(input_size, hidden_size, surrogate)
        self.hidden_size = hidden_size

    def forward(self, x_seq, seed_capacity, forecast_steps):
        batch = x_seq.size(0)
        h = torch.zeros(batch, self.hidden_size, device=x_seq.device)
        cap = seed_capacity.to(x_seq.device)
        preds = []
        for t in range(forecast_steps):
            inp = torch.cat([x_seq[:,t,:], cap], dim=1)
            h, drop = self.cell(inp, h)
            cap = cap - drop
            preds.append(cap.squeeze(1))
        return torch.stack(preds, dim=1)

class BaselineRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.fc    = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        h_next = self.activation(self.W_ih(x) + self.W_hh(h))
        drop   = self.fc(h_next)
        return h_next, drop

class BaselineMultiStepRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = BaselineRNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x_seq, seed_capacity, forecast_steps):
        batch = x_seq.size(0)
        h = torch.zeros(batch, self.hidden_size, device=x_seq.device)
        cap = seed_capacity.to(x_seq.device)
        preds = []
        for t in range(forecast_steps):
            inp = torch.cat([x_seq[:,t,:], cap], dim=1)
            h, drop = self.cell(inp, h)
            cap = cap - drop
            preds.append(cap.squeeze(1))
        return torch.stack(preds, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# 3) TRAINING RNNs & PREP DATA
# ──────────────────────────────────────────────────────────────────────────────
def run_training():
    # 3.1) train surrogate
    rf, scaler_sim = train_surrogate()

    # 3.2) load Batch 1 (test) & Batch 2 (train) data
    df1 = pd.read_pickle('Processed_data_Cycling&RPT_Batch1_Capacity_Forecasting_merged_update_Jan2025.pkl')
    df2 = pd.read_pickle('Processed_data_Cycling&RPT_Batch2_Capacity_Forecasting_merged_update_Jan2025.pkl')

    test_df  = df1[~df1['Group'].isin(['G12'])].dropna()
    train_df = df2[~df2['Group'].isin(['G11','G14'])].dropna()

    train_df = train_df[train_df['Cell'].isin(['C1','C3'])].copy()
    train_df['UID'] = train_df['Group'] + '-' + train_df['Cell']
    uids = train_df['UID'].unique().tolist()
    val_uids = random.sample(uids, 3)
    val_df = train_df[train_df['UID'].isin(val_uids)]
    tr_df  = train_df[~train_df['UID'].isin(val_uids)]

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

    # 3.3) scale
    scaler = MinMaxScaler().fit(tr_df[features].values)
    X_tr_s = scaler.transform(tr_df[features].values)
    X_va_s = scaler.transform(val_df[features].values)
    X_te_s = scaler.transform(test_df[features].values)

    y_tr = tr_df[target].values
    y_va = val_df[target].values
    y_te = test_df[target].values

    # 3.4) create sequences
    def create_seq(data, tgt, steps):
        X, y = [], []
        for i in range(len(data)-steps+1):
            X.append(data[i:i+steps])
            y.append(tgt[i:i+steps])
        return np.array(X), np.array(y)

    steps = 10
    X_tr, y_tr_seq = create_seq(X_tr_s, y_tr, steps)
    X_va, y_va_seq = create_seq(X_va_s, y_va, steps)
    X_te, y_te_seq = create_seq(X_te_s, y_te, steps)

    # 3.5) train loops
    input_size = X_tr.shape[2] + 1
    hidden_size = 50
    lr, epochs, patience = 0.001, 2500, 50

    # tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va_seq, dtype=torch.float32)

    # PI‑RNN
    pi_model = MultiStepPIRNN(input_size, hidden_size, rf)
    opt_pi   = optim.Adam(pi_model.parameters(), lr=lr)
    loss_fn  = nn.MSELoss()

    best_val, wait = float('inf'), 0
    for ep in range(1, epochs+1):
        pi_model.train()
        opt_pi.zero_grad()
        seed = y_tr_t[:,0].unsqueeze(1)
        out  = pi_model(X_tr_t, seed, steps)
        loss_tr = loss_fn(out, y_tr_t[:,:steps])
        loss_tr.backward()
        opt_pi.step()

        pi_model.eval()
        with torch.no_grad():
            seed_va = y_va_t[:,0].unsqueeze(1)
            out_va = pi_model(X_va_t, seed_va, steps)
            loss_va = loss_fn(out_va, y_va_t[:,:steps])

        if loss_va < best_val:
            best_val, wait = loss_va, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"PI‑RNN early stop at epoch {ep}")
                break
        if ep % 100 == 0:
            print(f"[PI] Epoch {ep}/{epochs} tr {loss_tr:.4f} va {loss_va:.4f}")

    # Baseline RNN
    base_model = BaselineMultiStepRNN(input_size, hidden_size)
    opt_b      = optim.Adam(base_model.parameters(), lr=lr)

    best_val, wait = float('inf'), 0
    for ep in range(1, epochs+1):
        base_model.train()
        opt_b.zero_grad()
        seed = y_tr_t[:,0].unsqueeze(1)
        out  = base_model(X_tr_t, seed, steps)
        loss_tr = loss_fn(out, y_tr_t[:,:steps])
        loss_tr.backward()
        opt_b.step()

        base_model.eval()
        with torch.no_grad():
            seed_va = y_va_t[:,0].unsqueeze(1)
            out_va = base_model(X_va_t, seed_va, steps)
            loss_va = loss_fn(out_va, y_va_t[:,:steps])

        if loss_va < best_val:
            best_val, wait = loss_va, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Base‑RNN early stop at epoch {ep}")
                break
        if ep % 100 == 0:
            print(f"[Base] Epoch {ep}/{epochs} tr {loss_tr:.4f} va {loss_va:.4f}")

    # return trained models & test sequences
    return pi_model, base_model, X_te, y_te_seq, steps
