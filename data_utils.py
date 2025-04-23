import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from models import train_pbm_surrogate_for_PI_RNN

def load_batch(path):
    """
    Load a pickled batch, sort it, and compute the per-step capacity drop.
    """
    df = pd.read_pickle(path)
    df.sort_values(['Channel Number','Group','Cell','RPT Number'], inplace=True)
    df['capacity_drop'] = (
        df.groupby(['Channel Number','Group','Cell'])['Capacity']
          .diff()
          .abs()
          .fillna(0)
    )
    return df

def prepare_pbm_surrogate(file_paths, sim_features, sim_target, seed=40):
    """
    Train the PBM surrogate (capacity-drop RandomForest).
    Returns (rf_model, scaler_sim).
    """
    from models import train_pbm_surrogate_for_PI_RNN
    rf_model, scaler_sim = train_pbm_surrogate_for_PI_RNN(
        file_paths, sim_features, sim_target, seed=seed
    )
    return rf_model, scaler_sim

def prepare_battery_sequences(
    batch1_path: str,
    batch2_path: str,
    features: list,
    target: str,
    val_cell_count: int = 3,
    seed: int = 40
):
    """
    Loads batch1 & batch2, filters by group & cell,
    splits training into train/val by Unique_Cell_ID,
    scales the features, and returns:
      X_train_s, y_train,
      X_val_s,   y_val,
      X_test_s,  y_test,
      scaler,    test_df
    """
    random.seed(seed)

    # Load & compute drops
    batch1 = load_batch(batch1_path)
    batch2 = load_batch(batch2_path)

    # Test = batch1 minus group G12
    test_df  = batch1[~batch1['Group'].isin(['G12'])].dropna()
    # Train+Val = batch2 minus groups G11, G14
    train_df = batch2[~batch2['Group'].isin(['G11','G14'])].dropna()

    # Only cells C1 & C3 for training/val
    train_df = train_df[train_df['Cell'].isin(['C1','C3'])].copy()

    # Split train/val by Unique_Cell_ID
    train_df['Unique_Cell_ID'] = train_df['Group'] + '-' + train_df['Cell']
    ids = train_df['Unique_Cell_ID'].unique().tolist()
    val_ids = random.sample(ids, val_cell_count)

    validation_df = train_df[train_df['Unique_Cell_ID'].isin(val_ids)]
    training_df   = train_df[~train_df['Unique_Cell_ID'].isin(val_ids)]

    # Scale
    scaler   = MinMaxScaler().fit(training_df[features])
    X_train_s = scaler.transform(training_df[features])
    y_train   = training_df[target].values

    X_val_s   = scaler.transform(validation_df[features])
    y_val     = validation_df[target].values

    X_test_s  = scaler.transform(test_df[features])
    y_test    = test_df[target].values

    return X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df

def create_sequences(arr: np.ndarray, vals: np.ndarray, steps: int):
    """
    Build overlapping sequences of length `steps` for RNN training.
    """
    Xs, ys = [], []
    for i in range(len(arr) - steps + 1):
        Xs.append(arr[i : i + steps])
        ys.append(vals[i : i + steps])
    return np.array(Xs), np.array(ys)
