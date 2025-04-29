import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from models import train_pbm_surrogate_for_PI_RNN

# —————————————————————————————
# 0. Dataset Configuration (all in one place)
# —————————————————————————————
# PBM pickles
PBM_SIM_PATHS = [
    'Simulated_PBM_data/G18_PBM_Simulated.pkl',
    'Simulated_PBM_data/G16_PBM_Simulated.pkl',
    'Simulated_PBM_data/G4_PBM_Simulated.pkl',
    'Simulated_PBM_data/G3_PBM_Simulated.pkl',
    'Simulated_PBM_data/G2_PBM_Simulated.pkl'
]
PBM_FEATURES = [
    'Ampere-Hour Throughput (Ah)',
    'Total Time Elapsed (h)',
    'Total Absolute Time From Start (h)',
    'Time Below 3A (h)',
    'Time Between 3A and 4A (h)',
    'Time Above 4A (h)',
    'RPT Number',
    'Capacity'
]
PBM_TARGET = 'Capacity_Drop_Ah'

# cycling & RPT data
BATCH1_PATH = 'Processed_data/Processed_data_Cycling&RPT_Batch1_Capacity_Forecasting_merged_update_Jan2025.pkl'
BATCH2_PATH = 'Processed_data/Processed_data_Cycling&RPT_Batch2_Capacity_Forecasting_merged_update_Jan2025.pkl'

BATTERY_FEATURES = [
    'Ampere-Hour Throughput (Ah)',
    'Total Absolute Time From Start (h)',
    'Total Time Elapsed (h)',
    'Time Below 3A (h)',
    'Time Between 3A and 4A (h)',
    'Time Above 4A (h)',
    'RPT Number'
]
BATTERY_TARGET = 'Capacity'

# group filters
TEST_EXCLUDE_GROUPS  = ['G12']
TRAIN_EXCLUDE_GROUPS = ['G11','G14']
TRAIN_CELLS          = ['C1','C3']

# validation split
VAL_CELL_COUNT = 3

# —————————————————————————————
# 1. PBM surrogate loader
# —————————————————————————————
def load_pbm_surrogate(seed: int = 40):
    """
    Trains the PBM surrogate model on all sim pickles.
    Returns (rf_model, scaler_sim, sim_df).
    """
    # load & concat
    sim_dfs = [pd.read_pickle(fp) for fp in PBM_SIM_PATHS]
    sim_df = pd.concat(sim_dfs, ignore_index=True)
    # train RF
    X = sim_df[PBM_FEATURES].values
    y = sim_df[PBM_TARGET].values
    scaler_sim = MinMaxScaler().fit(X)
    Xs = scaler_sim.transform(X)
    rf_model, _ = train_pbm_surrogate_for_PI_RNN(
        PBM_SIM_PATHS, PBM_FEATURES, PBM_TARGET, seed=seed
    )
    return rf_model, scaler_sim, sim_df

# —————————————————————————————
# 2. Batch loader + drop calc
# —————————————————————————————
def load_batch(path: str) -> pd.DataFrame:
    """
    Reads a pickle, sorts, and adds a 'capacity_drop' column.
    """
    df = pd.read_pickle(path)
    df = df.sort_values(['Channel Number','Group','Cell','RPT Number'])
    df['capacity_drop'] = (
        df
        .groupby(['Channel Number','Group','Cell'])['Capacity']
        .diff().abs().fillna(0)
    )
    return df

# —————————————————————————————
# 3. Battery data prep
# —————————————————————————————
def load_battery_data(seed: int = 40):
    """
    Loads batch1 & batch2, filters out excluded groups/cells,
    splits train/val by Unique_Cell_ID, scales features,
    and returns:
      X_train_s, y_train,
      X_val_s,   y_val,
      X_test_s,  y_test,
      scaler,    test_df
    """
    random.seed(seed)

    b1 = load_batch(BATCH1_PATH)
    b2 = load_batch(BATCH2_PATH)

    test_df = b1[~b1['Group'].isin(TEST_EXCLUDE_GROUPS)].dropna()
    tv = b2[~b2['Group'].isin(TRAIN_EXCLUDE_GROUPS)].dropna()
    tv = tv[tv['Cell'].isin(TRAIN_CELLS)].copy()

    tv['Unique_Cell_ID'] = tv['Group'] + '-' + tv['Cell']
    ids = tv['Unique_Cell_ID'].unique().tolist()
    val_ids = random.sample(ids, VAL_CELL_COUNT)

    val_df = tv[tv['Unique_Cell_ID'].isin(val_ids)]
    tr_df  = tv[~tv['Unique_Cell_ID'].isin(val_ids)]

    scaler   = MinMaxScaler().fit(tr_df[BATTERY_FEATURES])
    X_train_s = scaler.transform(tr_df[BATTERY_FEATURES])
    y_train   = tr_df[BATTERY_TARGET].values

    X_val_s   = scaler.transform(val_df[BATTERY_FEATURES])
    y_val     = val_df[BATTERY_TARGET].values

    X_test_s  = scaler.transform(test_df[BATTERY_FEATURES])
    y_test    = test_df[BATTERY_TARGET].values

    return X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df

# —————————————————————————————
# 4. Sequence builder for RNN
# —————————————————————————————
def make_sequences(
    X: np.ndarray, y: np.ndarray, steps: int
) -> tuple[np.ndarray,np.ndarray]:
    """
    Given scaled features X and targets y, produce
    overlapping sequences of length `steps`.
    """
    seqX, seqY = [], []
    for i in range(len(X) - steps + 1):
        seqX.append(X[i : i + steps])
        seqY.append(y[i : i + steps])
    return np.array(seqX), np.array(seqY)
