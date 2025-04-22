import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
import GPy

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
    def __init__(self):
        self.empirical_params = {}
        self.gpr_models = {}

    @staticmethod
    def empirical_model(x, a, b, c):
        return a + b * np.exp(c * x)

    def fit_empirical_model(self, X, y):
        popt, _ = curve_fit(
            self.empirical_model, 
            X.flatten(), 
            y.flatten(), 
            maxfev=10000, 
            p0=[np.mean(y), -1, -0.1]
        )
        return popt

    def fit(self, cell_data, cell_key, initial_points=6):
        cell_data = cell_data.sort_values('RPT Number')
        first_points = cell_data.iloc[:initial_points]

        X_train = first_points[['RPT Number']].values
        y_train = first_points[['Capacity']].values

        # Fit empirical model
        params = self.fit_empirical_model(X_train, y_train)
        self.empirical_params[cell_key] = params

        # Fit GPR on residuals
        y_empirical_train = self.empirical_model(X_train, *params)
        residuals_train = y_train - y_empirical_train

        kernel = GPy.kern.RBF(input_dim=1)
        gpr = GPy.models.GPRegression(X_train, residuals_train, kernel)
        gpr.optimize()

        self.gpr_models[cell_key] = gpr

    def predict(self, cell_data, cell_key, initial_points=6):
        if cell_key not in self.empirical_params or cell_key not in self.gpr_models:
            raise ValueError(f"No fitted model found for cell {cell_key}. Fit first.")

        cell_data = cell_data.sort_values('RPT Number')
        remaining_points = cell_data.iloc[initial_points:]

        X_test = remaining_points[['RPT Number']].values
        params = self.empirical_params[cell_key]
        y_empirical_test = self.empirical_model(X_test, *params)

        gpr = self.gpr_models[cell_key]
        residuals_pred, _ = gpr.predict(X_test)

        y_final_pred = y_empirical_test + residuals_pred
        y_true = remaining_points[['Capacity']].values.flatten()

        return y_true, y_final_pred.flatten()



class PBMSurrogate:
    def __init__(self, features, target, n_estimators=50, random_state=40):
        self.features = features
        self.target = target
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def load_simulation_data(self, file_paths):
        sim_dfs = [pd.read_pickle(fp) for fp in file_paths]
        sim_data_df = pd.concat(sim_dfs, ignore_index=True)
        return sim_data_df

    def fit(self, sim_data_df):
        X_sim = sim_data_df[self.features].values
        y_sim = sim_data_df[self.target].values
        X_sim_scaled = self.scaler.fit_transform(X_sim)
        self.model.fit(X_sim_scaled, y_sim)

    def predict(self, df):
        X_test = df[self.features].values
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def evaluate(self, df):
        y_true = df[self.target].values
        y_pred = self.predict(df)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse


