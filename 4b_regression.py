# interaction regression

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

# create interactions
X_train_interact = X_train.copy()
X_test_interact = X_test.copy()

X_train_interact['temp_precip'] = X_train['current_temperature'] * X_train['current_precipitation']
X_test_interact['temp_precip'] = X_test['current_temperature'] * X_test['current_precipitation']

X_train_interact['temp_humidity'] = X_train['current_temperature'] * X_train['current_specific_humidity']
X_test_interact['temp_humidity'] = X_test['current_temperature'] * X_test['current_specific_humidity']

X_train_interact['precip_humidity'] = X_train['current_precipitation'] * X_train['current_specific_humidity']
X_test_interact['precip_humidity'] = X_test['current_precipitation'] * X_test['current_specific_humidity']

# run regression ----
model_interact = LinearRegression()
model_interact.fit(X_train_interact, y_train)

y_pred_train_interact = model_interact.predict(X_train_interact)
y_pred_test_interact = model_interact.predict(X_test_interact)

# evaluate model ----
for feature, coef in zip(X_train_interact.columns, model_interact.coef_):
    print(f"{feature:25s}: {coef:.3f}")

print(f"Intercept: {model_interact.intercept_:.3f}")
print(f"Train R²: {r2_score(y_train, y_pred_train_interact):.3f}")
print(f"Test R²:  {r2_score(y_test, y_pred_test_interact):.3f}")

# visualize ----
# ensure arrays are aligned and clean
y_obs = y_test.to_numpy(dtype=float)
y_hat = y_pred_test_interact.astype(float)

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
    color='darkgreen',
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
plt.savefig("figures/4b_interaction_reg_viz.png", dpi=450, bbox_inches='tight')
plt.show()
