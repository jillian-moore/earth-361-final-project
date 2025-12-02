# lagged 12-week avg regression

# load packages ----
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# prepare features ----
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

for feature, coef in zip(features, model.coef_):
    print(f"{feature:35s}: {coef:.2f}")
print(f"{'Intercept':35s}: {model.intercept_:.2f}")