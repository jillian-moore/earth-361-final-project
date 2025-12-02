# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

print(f"Data shape: {fever_df.shape}")
print(f"Variables: {fever_df.columns.tolist()}")

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

# ============================================
# MODEL 1: SIMPLE LINEAR REGRESSION
# ============================================
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

y_pred_train_linear = model_linear.predict(X_train)
y_pred_test_linear = model_linear.predict(X_test)

print("\nLinear Model Coefficients:")
for feature, coef in zip(features, model_linear.coef_):
    print(f"{feature:25s}: {coef:.3f}")
print(f"Intercept: {model_linear.intercept_:.3f}")

print(f"Train R²: {r2_score(y_train, y_pred_train_linear):.3f}")
print(f"Test R²:  {r2_score(y_test, y_pred_test_linear):.3f}")

# ============================================
# MODEL 2: INTERACTION REGRESSION
# ============================================
X_train_interact = X_train.copy()
X_test_interact = X_test.copy()

# create interactions
X_train_interact['temp_precip'] = X_train['current_temperature'] * X_train['current_precipitation']
X_test_interact['temp_precip'] = X_test['current_temperature'] * X_test['current_precipitation']

X_train_interact['temp_humidity'] = X_train['current_temperature'] * X_train['current_specific_humidity']
X_test_interact['temp_humidity'] = X_test['current_temperature'] * X_test['current_specific_humidity']

X_train_interact['precip_humidity'] = X_train['current_precipitation'] * X_train['current_specific_humidity']
X_test_interact['precip_humidity'] = X_test['current_precipitation'] * X_test['current_specific_humidity']

model_interact = LinearRegression()
model_interact.fit(X_train_interact, y_train)

y_pred_test_interact = model_interact.predict(X_test_interact)
print(f"Interaction Model Test R²: {r2_score(y_test, y_pred_test_interact):.3f}")

# ============================================
# MODEL 3: LOG-TRANSFORMED REGRESSION
# ============================================
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

y_pred_test_log = np.expm1(model_log.predict(X_test))
print(f"Log-Transform Model Test R²: {r2_score(y_test, y_pred_test_log):.3f}")

# ============================================
# MODEL 4: POLYNOMIAL REGRESSION
# ============================================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

model_poly = LinearRegression()
model_poly.fit(X_train_poly_scaled, y_train)

y_pred_test_poly = model_poly.predict(X_test_poly_scaled)
print(f"Polynomial Model Test R²: {r2_score(y_test, y_pred_test_poly):.3f}")

# ============================================
# MODEL COMPARISON SUMMARY
# ============================================
results = pd.DataFrame({
    'Model': ['Linear', 'Interaction', 'Log-Transform', 'Polynomial'],
    'Test_R2': [
        r2_score(y_test, y_pred_test_linear),
        r2_score(y_test, y_pred_test_interact),
        r2_score(y_test, y_pred_test_log),
        r2_score(y_test, y_pred_test_poly)
    ]
})

print(results)
results.to_csv('output/fever_model_comparison.csv', index=False)


# ============================================
# AUTOREGRESSIVE REGRESSION MODEL
# ============================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ------------------------------
# Load data
# ------------------------------
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# Sort by city and date to preserve temporal order
fever_df['date'] = pd.to_datetime(fever_df['date'])
fever_df = fever_df.sort_values(['city', 'date'])

# ------------------------------
# Create lagged features
# ------------------------------
lags = [1, 2, 3]  # you can adjust number of lags
for lag in lags:
    fever_df[f'cases_lag_{lag}'] = fever_df.groupby('city')['total_cases'].shift(lag)

# ------------------------------
# Select predictors and target
# ------------------------------
features = [
    'cases_lag_1', 'cases_lag_2', 'cases_lag_3',
    'current_temperature', 'current_precipitation', 'current_specific_humidity',
    'vegetation_ne', 'vegetation_nw', 'vegetation_se', 'vegetation_sw'
]
target = 'total_cases'

# Drop rows with NaNs from lagging
model_data = fever_df[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

print(f"Model data shape: {X.shape}")

# ------------------------------
# Train-test split (time-aware)
# ------------------------------
# Use 80% of time series for training, last 20% for testing
split_idx = int(len(model_data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ------------------------------
# Fit linear regression
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ------------------------------
# Evaluate model
# ------------------------------
def print_metrics(y_true, y_pred, dataset=""):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{dataset} R²: {r2:.3f}, RMSE: {rmse:.1f}, MAE: {mae:.1f}")

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test, y_pred_test, "Test")

# Print coefficients
print("\nCoefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature:25s}: {coef:.2f}")
print(f"{'Intercept':25s}: {model.intercept_:.2f}")

# ------------------------------
# Plot predicted vs actual
# ------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Total Cases')
plt.ylabel('Predicted Total Cases')
plt.title('Autoregressive Model: Predicted vs Actual')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ENVIRONMENTAL PART 2
 # ============================
# Compute 12-week rolling averages per city
# ============================
climate_vars = [
    'current_temperature', 
    'current_precipitation', 
    'current_specific_humidity', 
    'current_max_temperature', 
    'current_min_temperature', 
    'current_diurnal_temperature_range'
]

for var in climate_vars:
    colname = f"{var}_12wk_avg"
    fever_df[colname] = (
        fever_df.groupby('city')[var]
        .rolling(window=12, min_periods=1)
        .mean()
        .shift(1)  # lag so we only use past data
        .reset_index(0, drop=True)
    )

# ============================
# Select features
# ============================
static_features = ['city', 'year', 'vegetation_ne', 'vegetation_nw', 'vegetation_se', 'vegetation_sw']

rolling_features = [f"{var}_12wk_avg" for var in climate_vars]

features = static_features + rolling_features
target = 'total_cases'

# Drop rows with missing values (from rolling windows)
model_data = fever_df[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

# ============================
# Train-test split
# ============================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================
# Fit Linear Regression
# ============================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ============================
# Evaluation function
# ============================
def print_metrics(y_true, y_pred, label=""):
    print(f"{label} R²: {r2_score(y_true, y_pred):.3f}")
    print(f"{label} RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"{label} MAE: {mean_absolute_error(y_true, y_pred):.2f}")

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test, y_pred_test, "Test")

# ============================
# Coefficients
# ============================
print("\nCoefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature:35s}: {coef:.2f}")
print(f"{'Intercept':35s}: {model.intercept_:.2f}")

