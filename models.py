import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

def train_pbm_surrogate(file_paths, sim_features, sim_target, seed=40):
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