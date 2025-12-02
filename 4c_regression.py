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
def print_metrics(y_true, y_pred, dataset=""):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{dataset} R²: {r2:.3f}, RMSE: {rmse:.1f}, MAE: {mae:.1f}")

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test, y_pred_test, "Test")

for feature, coef in zip(features, model.coef_):
    print(f"{feature:25s}: {coef:.2f}")
print(f"{'Intercept':25s}: {model.intercept_:.2f}")
