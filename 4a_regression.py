# simple linear regression

# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# prepare features ----
features = ['current_temperature', 'current_precipitation', 'current_specific_humidity', 
            'city', 'year', 'vegetation_ne', 'vegetation_nw', 'vegetation_se', 'vegetation_sw',
            'current_max_temperature', 'current_min_temperature',
            'current_diurnal_temperature_range']
target = 'total_cases'

# remove missing values
model_data = fever_df[features + [target]].dropna()

# split train/test ----
X = model_data[features]
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# run regression ----
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

y_pred_train_linear = model_linear.predict(X_train)
y_pred_test_linear = model_linear.predict(X_test)

# evaluate model ----
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_interact))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_interact))
train_mae = mean_absolute_error(y_train, y_pred_train_interact)
test_mae = mean_absolute_error(y_test, y_pred_test_interact)
train_r2 = r2_score(y_train, y_pred_train_interact)
test_r2 = r2_score(y_test, y_pred_test_interact)

# results dict ----
results = {
    'model_type': 'Simple Linear Regression',  
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2)
}

# save results ----
results_df = pd.DataFrame([results])
results_df.to_csv("results/simple_reg_results_table.csv", index=False)

# save predictions ----
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test_interact
})
predictions_df.to_csv('results/simple_reg_predictions.csv', index=False)

# visualize ----
# ensure arrays are aligned and clean
y_obs = y_test.to_numpy(dtype=float)
y_hat = y_pred_test_linear.astype(float)

# remove any NaN pairs
mask = ~np.isnan(y_obs) & ~np.isnan(y_hat)
y_obs = y_obs[mask]
y_hat = y_hat[mask]

# compute 1:1 line limits
lims = [min(y_obs.min(), y_hat.min()),
        max(y_obs.max(), y_hat.max())]

# format
title_size = 22
label_size = 20
tick_size  = 16

plt.figure(figsize=(8, 7))

# scatter plot
plt.scatter(
    y_obs, y_hat,
    color='steelblue',
    alpha=0.6,
    s=40,
    edgecolor='none'
)

# 1:1 reference line
plt.plot(lims, lims, 'r--', linewidth=2.5)

# titles + labels
plt.title("Observed vs Predicted Dengue Cases", fontsize=title_size)
plt.xlabel("Observed Cases", fontsize=label_size)
plt.ylabel("Predicted Cases", fontsize=label_size)
plt.tick_params(axis='both', labelsize=tick_size)

# grid + layout
plt.grid(alpha=0.3)
plt.xlim(lims)
plt.ylim(lims)

plt.tight_layout()
plt.savefig("figures/4a_reg_viz.png", dpi=450, bbox_inches='tight')
plt.show()
