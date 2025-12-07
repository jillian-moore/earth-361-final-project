# autoregressive regression

# load packages ----
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# sort by city and date to preserve temporal order
fever_df['date'] = pd.to_datetime(fever_df['date'])
fever_df = fever_df.sort_values(['city', 'date'])

# create lagged features ----
lags = [1, 2, 3] 
for lag in lags:
    fever_df[f'cases_lag_{lag}'] = fever_df.groupby('city')['total_cases'].shift(lag)

# prepare features ----
features = [
    'cases_lag_1', 'cases_lag_2', 'cases_lag_3',
    'current_temperature', 'current_precipitation', 'current_specific_humidity',
    'vegetation_ne', 'vegetation_nw', 'vegetation_se', 'vegetation_sw'
]
target = 'total_cases'

# drop rows with NaNs from lagging
model_data = fever_df[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

# split train/test ----
split_idx = int(len(model_data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# run regression ----
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# evaluate model ----
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# results dict
results = {
    'model_type': 'Autoregressive Regression',  
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2)
}

results_df = pd.DataFrame([results])
results_df.to_csv("results/auto_reg_results_table.csv", index=False)

# save predictions ----
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test
})
predictions_df.to_csv('results/auto_reg_results_predictions.csv', index=False)

# visualize ----
# ensure arrays are aligned and clean
y_obs = y_test.to_numpy(dtype=float)
y_hat = y_pred_test.astype(float)

# remove any NaN pairs (just in case)
mask = ~np.isnan(y_obs) & ~np.isnan(y_hat)
y_obs = y_obs[mask]
y_hat = y_hat[mask]

# compute 1:1 line limits
lims = [min(y_obs.min(), y_hat.min()),
        max(y_obs.max(), y_hat.max())]

# formatting
title_size = 22
label_size = 20
tick_size  = 16

plt.figure(figsize=(8, 7))

# scatter plot
plt.scatter(
    y_obs, y_hat,
    color='purple',
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
plt.savefig("figures/4c_autoreg_viz.png", dpi=450, bbox_inches='tight')
plt.show()
