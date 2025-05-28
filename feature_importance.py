# feature_analysis.py

import random
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

from data_utils import load_battery_data, BATTERY_FEATURES, make_sequences
import data_utils

# —————————————————————————————
# 0. Settings & Reproducibility
# —————————————————————————————
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]   = 16

SEED = 40
random.seed(SEED)
np.random.seed(SEED)

# Use 'capacity_drop' as target
data_utils.BATTERY_TARGET = "capacity_drop"
forecast_steps = 1

# Select features to use
selected_features = [f for f in BATTERY_FEATURES if f != "RPT Number"]

# Pretty names for SHAP plot
new_feature_names = [
    'Ampere-Hour Throughput (Ah)', 
    'Total Time Elapsed From Start (h)', 
    'Time Under Load (h)',  
    'Time Duration Below 3A (h)', 
    'Time Duration Between 3A and 4A (h)',  
    'Time Duration Above 4A (h)'
]
feature_mapping = dict(zip(selected_features, new_feature_names))
feature_indices = [selected_features.index(f) for f in selected_features]

# —————————————————————————————
# 1. Load and preprocess data
# —————————————————————————————
X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df = load_battery_data(seed=SEED)

# Sequence targets
_, y_train_seq = make_sequences(X_train_s, y_train, forecast_steps)
_, y_test_seq  = make_sequences(X_test_s,  y_test,  forecast_steps)
y_train_seq = y_train_seq[:, 0]
y_test_seq  = y_test_seq[:, 0]

# Inverse transform features for interpretability
X_train_orig = scaler.inverse_transform(X_train_s)
X_test_orig  = scaler.inverse_transform(X_test_s)

X_train_df = pd.DataFrame(X_train_orig[:len(y_train_seq)][:, feature_indices], columns=selected_features)
X_test_df  = pd.DataFrame(X_test_orig[:len(y_test_seq)][:, feature_indices],  columns=selected_features)

# —————————————————————————————
# 2. SHAP Feature Importance with LightGBM
# —————————————————————————————
model = lgb.LGBMRegressor()
model.fit(X_train_df, y_train_seq)

explainer = shap.Explainer(model, X_train_df)
shap_values = explainer(X_test_df)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': [feature_mapping[feat] for feat in selected_features],
    'Mean Absolute SHAP Value': mean_abs_shap
}).sort_values('Mean Absolute SHAP Value', ascending=False)

# Plot bar chart
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
plt.xticks([0,0.005,0.01,0.015,0.02,0.025], fontsize=14)
plt.xlim([0, 0.025])
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
