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

# linear model coefs ----
for feature, coef in zip(features, model_linear.coef_):
    print(f"{feature:25s}: {coef:.3f}")

print(f"Intercept: {model_linear.intercept_:.3f}")
print(f"Train R²: {r2_score(y_train, y_pred_train_linear):.3f}")
print(f"Test R²:  {r2_score(y_test, y_pred_test_linear):.3f}")