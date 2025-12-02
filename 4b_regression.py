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
