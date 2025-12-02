# elastic net regression

# load packages ----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# load data ----
df = pd.read_csv('data/processed/dengue_data_cleaned.csv')

# define features (exclude target and non-predictive columns) ----
exclude_cols = ['total_cases', 'date', 'is_outbreak', 'log_cases', 
                'cases_zscore', 'cases_lag', 'cases_3mo_avg']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# prepare data ----
X = df[feature_cols]
y = df['total_cases']

# encode categorical variables ----
categorical_cols = ['city', 'season', 'temp_category', 'rain_category', 'humidity_category']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# split data ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define repeated k-fold cross-validation ----
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# define hyperparameter grid ----
param_grid = {
    'alpha': [0.01, 0.1, 1, 10],        # regularization strength
    'l1_ratio': [0.1, 0.5, 0.9]         # mixing parameter (L1 vs L2)
}

# initialize base model ----
base_model = ElasticNet(random_state=42, max_iter=5000)

# perform grid search with cross-validation ----
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=rkf,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# get best model ----
model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_score = -grid_search.best_score_

# predictions ----
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# evaluate ----
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# results ----
results = {
    'model_type': 'Elastic Net Regression',
    'task': 'regression',
    'best_params': best_params,
    'cv_config': {'n_splits': 5, 'n_repeats': 3},
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'cv_rmse': float(best_cv_score),
    'feature_importance': dict(zip(X.columns, model.coef_.tolist()))
}

print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test R²: {test_r2:.4f}")
print(f"CV RMSE: {best_cv_score:.2f}")

# save model and results ----
results_df = pd.DataFrame([results])
results_df.to_csv("results/elasticnet_results_table.csv", index=False)
results_df.to_latex("results/elasticnet_results_table.tex", index=False, float_format="%.3f")

# predictions ----
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test
})
predictions_df.to_csv('results/elasticnet_predictions.csv', index=False)

# visualizations ----
# actual vs predicted
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Cases")
plt.ylabel("Predicted Cases")
plt.title("Elastic Net: Actual vs Predicted Dengue Cases")
plt.tight_layout()
plt.savefig("results/elasticnet_actual_vs_predicted.png", dpi=300)
plt.show()

# residuals plot
residuals = y_test - y_pred_test
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_pred_test, y=residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Cases")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("results/elasticnet_residual_plot.png", dpi=300)
plt.show()

# feature importance plot
coef_series = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
plt.figure(figsize=(8, 6))
coef_series.head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Coefficients (Elastic Net)")
plt.tight_layout()
plt.savefig("results/elasticnet_feature_importance.png", dpi=300)
plt.show()
