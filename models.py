import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
import GPy


# --------------------------
# 1. Train PBM Surrogate (for PI-RNN)
# Use 'train_pbm_surrogate_for_PI_RNN' to obtain a RandomForest surrogate (rf_model) and its scaler (scaler_sim).
# This surrogate (rf_model, scaler_sim) is distinct from the PBMSurrogate class below and is used
# internally to inject physics-based predictions into the PI-RNN model during training.

def train_pbm_surrogate_for_PI_RNN(file_paths, sim_features, sim_target, seed=40):
    """
    Load simulation data, fit a RandomForest surrogate.
    Returns: (surrogate_model, scaler_sim)
    """
    sim_dfs = [pd.read_pickle(fp) for fp in file_paths]
    sim_df = pd.concat(sim_dfs, ignore_index=True)
    X = sim_df[sim_features].values
    y = sim_df[sim_target].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(X_scaled, y)
    return rf, scaler


class CustomRNNCellWithSurrogate(nn.Module):
    def __init__(self, input_size, hidden_size, surrogate_model):
        super(CustomRNNCellWithSurrogate, self).__init__()
        self.input_size = input_size  
        self.hidden_size = hidden_size
        self.surrogate_model = surrogate_model
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pbm_weight = nn.Parameter(torch.randn(1, hidden_size))
        self.fc = nn.Linear(hidden_size, 1)  # Predict capacity drop

    def forward(self, x, hidden):
        raw_feat_count = self.input_size
        pbm_features = x[:, :raw_feat_count].detach().cpu().numpy()
        pbm_output = self.surrogate_model.predict(pbm_features)
        pbm_output = torch.tensor(pbm_output, dtype=torch.float32, device=x.device)
        pbm_output = pbm_output.unsqueeze(1).expand(-1, self.hidden_size)
        h_pbm = self.pbm_weight * pbm_output
        h_next = self.activation(self.W_ih(x) + self.W_hh(hidden) + h_pbm)
        capacity_drop = self.fc(h_next)
        return h_next, capacity_drop

class MultiStepPIRNN(nn.Module):
    def __init__(self, input_size, hidden_size, surrogate_model):
        super(MultiStepPIRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = CustomRNNCellWithSurrogate(input_size, hidden_size, surrogate_model)
        self.input_size = input_size

    def forward(self, x, current_capacity, forecast_steps):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        next_capacity = current_capacity.clone().to(x.device)
        all_predictions = []
        for t in range(forecast_steps):
            current_input = torch.cat((x[:, t, :], next_capacity), dim=1)
            h, capacity_drop = self.rnn_cell(current_input, h)
            next_capacity = next_capacity - capacity_drop
            all_predictions.append(next_capacity.squeeze(-1))
        return torch.stack(all_predictions, dim=1)


class BaselineRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaselineRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        h_next = self.activation(self.W_ih(x) + self.W_hh(hidden))
        capacity_drop = self.fc(h_next)
        return h_next, capacity_drop

class BaselineMultiStepRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaselineMultiStepRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = BaselineRNNCell(input_size, hidden_size)
        self.input_size = input_size

    def forward(self, x, current_capacity, forecast_steps):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        next_capacity = current_capacity.clone().to(x.device)
        predictions = []
        for t in range(forecast_steps):
            current_input = torch.cat((x[:, t, :], next_capacity), dim=1)
            h, capacity_drop = self.rnn_cell(current_input, h)
            next_capacity = next_capacity - capacity_drop
            predictions.append(next_capacity.squeeze(-1))
        return torch.stack(predictions, dim=1)
    


class GPRBaseline:
    def __init__(self, initial_points=9):
        # number of initial points to fit the empirical + GPR
        self.initial_points   = initial_points  
        self.empirical_params = {}
        self.gpr_models       = {}

    @staticmethod
    def empirical_model(x, a, b, c):
        return a + b * np.exp(c * x)

    def fit(self, cell_data, cell_key, initial_points=None):
        """Fit empirical + GPR on the first `initial_points` cycles."""
        ip = initial_points or self.initial_points
        cell_data = cell_data.sort_values('RPT Number')
        train_df  = cell_data.iloc[:ip]

        X_train = train_df[['RPT Number']].values.flatten()
        y_train = train_df['Capacity'].values

        # 1) empirical fit on first ip points
        popt, _ = curve_fit(
            self.empirical_model,
            X_train, y_train,
            p0=[np.mean(y_train), -1, -0.1],
            bounds=([-10, -10, -10], [10, 10, 10]),
            maxfev=10000
        )
        self.empirical_params[cell_key] = popt

        # 2) fit GPR on residuals of those ip points
        y_emp_train = self.empirical_model(X_train, *popt)
        resid_train = (y_train - y_emp_train).reshape(-1, 1)

        kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
        gpr   = GPy.models.GPRegression(
                    X_train.reshape(-1,1), resid_train, kernel
                )
        gpr.optimize()

        self.gpr_models[cell_key] = gpr

    def predict(self, cell_data, cell_key, initial_points=None):
        """Return full-length forecast beyond initial_points."""
        ip = initial_points or self.initial_points
        cell_data      = cell_data.sort_values('RPT Number')
        forecast_df    = cell_data.iloc[ip:]
        rpt_vals       = forecast_df['RPT Number'].values.reshape(-1,1)
        y_true         = forecast_df['Capacity'].values

        popt = self.empirical_params[cell_key]
        y_emp = self.empirical_model(rpt_vals.flatten(), *popt)
        gpr   = self.gpr_models[cell_key]
        resid_pred, _ = gpr.predict(rpt_vals)

        y_pred = y_emp + resid_pred.flatten()
        return y_true, y_pred

    def predict_horizon(self, cell_data, cell_key, steps, initial_points=None):
        """
        Return only the first `steps` points from predict().
        e.g. for steps=3, returns y_true[:3], y_pred[:3].
        """
        y_true_full, y_pred_full = self.predict(
            cell_data, cell_key, initial_points
        )
        # if fewer than `steps` points exist, it simply returns all of them
        return y_true_full[:steps], y_pred_full[:steps]


# models.py (excerpt)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

class PBMSurrogate:
    def __init__(
        self,
        features,
        capacity_target="Capacity",
        drop_target="Capacity_Drop_Ah",
        n_estimators=50,
        random_state=40
    ):
        self.features        = features
        self.capacity_target = capacity_target
        self.drop_target     = drop_target

        # single-step capacity surrogate
        self.scaler_cap = MinMaxScaler()
        self.model_cap  = RandomForestRegressor(
                              n_estimators=n_estimators,
                              random_state=random_state
                          )

        # single-step drop surrogate
        self.scaler_drop = MinMaxScaler()
        self.model_drop  = RandomForestRegressor(
                              n_estimators=n_estimators,
                              random_state=random_state
                          )

        # containers for multi-step drop surrogates
        self.scalers_h   = {}  # horizon h → MinMaxScaler
        self.models_h    = {}  # horizon h → RandomForestRegressor

    def load_simulation_data(self, file_paths):
        sim_dfs = [pd.read_pickle(fp) for fp in file_paths]
        return pd.concat(sim_dfs, ignore_index=True)

    # -- Single-step capacity model --
    def fit_capacity(self, sim_df):
        X = sim_df[self.features].values
        y = sim_df[self.capacity_target].values
        Xs = self.scaler_cap.fit_transform(X)
        self.model_cap.fit(Xs, y)

    def predict_capacity(self, df):
        X  = df[self.features].values
        Xs = self.scaler_cap.transform(X)
        return self.model_cap.predict(Xs)

    # -- Single-step drop model --
    def fit_drop(self, sim_df):
        X = sim_df[self.features].values
        y = sim_df[self.drop_target].values
        Xs = self.scaler_drop.fit_transform(X)
        self.model_drop.fit(Xs, y)

    def predict_drop(self, df):
        X  = df[self.features].values
        Xs = self.scaler_drop.transform(X)
        return self.model_drop.predict(Xs)

    # -- Multi-step drop surrogate (horizon‐specific) --
    def fit_horizon(self, sim_df, h):
        Xh, yh = [], []
        for i in range(len(sim_df) - h):
            block = sim_df[self.features].iloc[i : i+h].values.flatten()
            Xh.append(block)
            yh.append(sim_df[self.drop_target].iloc[i + h])
        Xh = np.vstack(Xh); yh = np.array(yh)

        scaler_h = MinMaxScaler().fit(Xh)
        model_h  = RandomForestRegressor(
                       n_estimators=self.model_drop.n_estimators,
                       random_state=self.model_drop.random_state
                   )
        model_h.fit(scaler_h.transform(Xh), yh)

        self.scalers_h[h] = scaler_h
        self.models_h[h]  = model_h

    def predict_capacity_multi(self, df, steps):
        """
        Recursive multi-step capacity forecast:
        capacity_pred[t+steps] = capacity_true[t] - sum_{j=0..steps-1} drop_pred[t+j]
        Returns a length-(len(df)-steps) array aligned to df.Capacity.values[steps:].
        """
        caps = df[self.capacity_target].values
        n    = len(df)
        y_pred = []
        for i in range(steps, n):
            cap_est = caps[i-steps]
            for j in range(steps):
                row = df[self.features].iloc[i-steps+j : i-steps+j+1]
                cap_est -= self.model_drop.predict(self.scaler_drop.transform(row))
            y_pred.append(cap_est)
        return np.array(y_pred)




