# lagged 12-week average regression

# load packages ----
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# prepare climate features ----
climate_vars = [
    'current_temperature', 
    'current_precipitation', 
    'current_specific_humidity', 
    'current_max_temperature', 
    'current_min_temperature', 
    'current_diurnal_temperature_range'
]

# compute 12-week rolling averages per city ----
for var in climate_vars:
    colname = f"{var}_12wk_avg"
    fever_df[colname] = (
        fever_df.groupby('city')[var]
        .rolling(window=12, min_periods=1)
        .mean()
        .shift(1)  # lag so we only use past data
        .reset_index(0, drop=True)
    )

# define predictors and target ----
static_features = ['city', 'year', 'vegetation_ne', 'vegetation_nw', 'vegetation_se', 'vegetation_sw']
rolling_features = [f"{var}_12wk_avg" for var in climate_vars]
features = static_features + rolling_features
target = 'total_cases'

# drop rows with missing values from rolling
model_data = fever_df[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

# split train/test ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# run regression ----
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# evaluate model ----
def print_metrics(y_true, y_pred, label=""):
    print(f"{label} R²: {r2_score(y_true, y_pred):.3f}")
    print(f"{label} RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"{label} MAE: {mean_absolute_error(y_true, y_pred):.2f}")

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test, y_pred_test, "Test")

# print coefficients
for feature, coef in zip(features, model.coef_):
    print(f"{feature:35s}: {coef:.2f}")
print(f"{'Intercept':35s}: {model.intercept_:.2f}")

# visualize ----
# ensure arrays are aligned and clean
y_obs = y_test.to_numpy(dtype=float)
y_hat = y_pred_test.astype(float)

# remove any NaN pairs
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
    color='hotpink', 
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
plt.savefig("figures/4d_lagged12wk_viz.png", dpi=450, bbox_inches='tight')
plt.show()
