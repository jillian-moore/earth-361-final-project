# interaction term regression

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

# create interactions ----
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

# predictions ----
y_pred_train_interact = model_interact.predict(X_train_interact)
y_pred_test_interact = model_interact.predict(X_test_interact)

# evaluate model ----
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_interact))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_interact))
train_mae = mean_absolute_error(y_train, y_pred_train_interact)
test_mae = mean_absolute_error(y_test, y_pred_test_interact)
train_r2 = r2_score(y_train, y_pred_train_interact)
test_r2 = r2_score(y_test, y_pred_test_interact)

# results dict ----
results = {
    'model_type': 'Interaction Term Regression',  
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2)
}

# save results ----
results_df = pd.DataFrame([results])
results_df.to_csv("results/interaction_reg_results_table.csv", index=False)

# save predictions ----
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test_interact
})
predictions_df.to_csv('results/interaction_reg_predictions.csv', index=False)

# get coefficients for interaction terms ----
coef_df = pd.DataFrame({
    'feature': X_train_interact.columns,
    'coefficient': model_interact.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features (by coefficient magnitude):")
print(coef_df.head(10).to_string(index=False))

# save coefficients ----
coef_df.to_csv('results/interaction_reg_coefficients.csv', index=False)

# visualization: observed vs predicted ----
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test_interact, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Observed Cases', fontsize=12)
plt.ylabel('Predicted Cases', fontsize=12)
plt.title(f'Interaction Term Regression: Observed vs Predicted\nTest R² = {test_r2:.3f}', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/interaction_reg_obs_vs_pred.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFiles saved:")
print("  - results/interaction_reg_results_table.csv")
print("  - results/interaction_reg_predictions.csv")
print("  - results/interaction_reg_coefficients.csv")
print("  - figures/interaction_reg_obs_vs_pred.png")