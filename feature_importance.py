"""
This script generates feature importance summary plot using SHAP analysis. 
"""

import random
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from data_utils import load_batch, BATCH1_PATH, BATCH2_PATH, BATTERY_FEATURES

# —————————————————————————————
# 0. Plotting and Seed Settings
# —————————————————————————————
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

SEED = 40
random.seed(SEED)
np.random.seed(SEED)

# —————————————————————————————
# 1. Load and Prepare Input Features
# —————————————————————————————
def prepare_training_data(seed=40):
    # Load raw batches
    df1 = load_batch(BATCH1_PATH)
    df2 = load_batch(BATCH2_PATH)

    # Filter test/train based on group criteria
    test_df = df1[~df1['Group'].isin(['G12'])].dropna()
    train_df = df2[~df2['Group'].isin(['G11', 'G14'])].dropna()

    # Restrict to cells C1 and C3
    train_df = train_df[train_df['Cell'].isin(['C1', 'C3'])].copy()
    train_df["Unique_Cell_ID"] = train_df["Group"] + "-" + train_df["Cell"]

    unique_ids = train_df["Unique_Cell_ID"].unique()
    random.seed(seed)
    val_ids = random.sample(list(unique_ids), 3)

    # Return only training portion
    training_df = train_df[~train_df["Unique_Cell_ID"].isin(val_ids)].copy()
    return training_df, test_df

training_df, test_df = prepare_training_data(seed=SEED)

# —————————————————————————————
# 2. Feature Selection
# —————————————————————————————
original_features = [f for f in BATTERY_FEATURES if f != "RPT Number"]

# Feature names for SHAP plot
pretty_names = [
    'Ampere-Hour Throughput (Ah)', 
    'Total Time Elapsed From Start (h)', 
    'Time Under Load (h)',  
    'Time Duration Below 3A (h)', 
    'Time Duration Between 3A and 4A (h)',  
    'Time Duration Above 4A (h)'
]
feature_mapping = dict(zip(original_features, pretty_names))

X_train = training_df[original_features]
y_train = training_df['capacity_drop']
X_test  = test_df[original_features]

# —————————————————————————————
# 3. Train Model and Compute SHAP
# —————————————————————————————
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': [feature_mapping[f] for f in original_features],
    'Mean Absolute SHAP Value': mean_abs_shap
}).sort_values('Mean Absolute SHAP Value', ascending=False)

# —————————————————————————————
# 4. Plot SHAP Summary
# —————————————————————————————
plt.figure(figsize=(10, 3))
colors = plt.cm.Purples(np.linspace(0.8, 0.4, len(feature_importance)))

plt.barh(
    feature_importance['Feature'], 
    feature_importance['Mean Absolute SHAP Value'], 
    color=colors, 
    edgecolor='black', 
    alpha=0.9
)
plt.gca().invert_yaxis()
plt.xlabel('Mean Absolute SHAP Value (-)', fontsize=16, fontweight='bold')
plt.xticks([0, 0.005, 0.01, 0.015, 0.02, 0.025], fontsize=14)
plt.xlim([0, 0.025])
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
