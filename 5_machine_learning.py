# machine learning pipeline for dengue prediction

# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            classification_report, confusion_matrix, roc_auc_score)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# load data ----
df = pd.read_csv("data/processed/dengue_data_cleaned.csv", index_col=0)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ============================================
# FEATURE ENGINEERING
# ============================================

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# encode categorical variables
le_district = LabelEncoder()
le_province = LabelEncoder()
le_season = LabelEncoder()

df['district_encoded'] = le_district.fit_transform(df['district'])
df['province_encoded'] = le_province.fit_transform(df['province'])
df['season_encoded'] = le_season.fit_transform(df['season'])

# create additional temporal features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# select features for modeling
numeric_features = [
    # temporal
    'year', 'month', 'quarter', 'month_sin', 'month_cos',
    
    # climate
    'temp_avg', 'precipitation_avg', 'humidity_avg', 'elevation',
    
    # engineered
    'temp_precip_interaction',
    'cases_lag_district', 'cases_lag_province', 'cases_3mo_avg',
    
    # encoded categorical
    'district_encoded', 'province_encoded', 'season_encoded', 'is_monsoon'
]

# remove rows with missing lagged features
df_model = df[numeric_features + ['cases', 'is_outbreak']].dropna()

print(f"\nFinal dataset for modeling: {df_model.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ============================================
# TASK 1: REGRESSION - PREDICT CASE COUNTS
# ============================================

print("\n" + "="*60)
print("TASK 1: REGRESSION - PREDICTING DENGUE CASE COUNTS")
print("="*60)

# prepare data
X = df_model[numeric_features]
y = df_model['cases']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# ============================================
# MODEL 1: LINEAR REGRESSION
# ============================================

print("\n" + "-"*60)
print("Model 1: Linear Regression")
print("-"*60)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr_train = lr.predict(X_train_scaled)
y_pred_lr_test = lr.predict(X_test_scaled)

print(f"Train R²: {r2_score(y_train, y_pred_lr_train):.3f}")
print(f"Test R²:  {r2_score(y_test, y_pred_lr_test):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr_test)):.1f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_lr_test):.1f}")

# ============================================
# MODEL 2: RIDGE REGRESSION
# ============================================

print("\n" + "-"*60)
print("Model 2: Ridge Regression (L2 Regularization)")
print("-"*60)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge_train = ridge.predict(X_train_scaled)
y_pred_ridge_test = ridge.predict(X_test_scaled)

print(f"Train R²: {r2_score(y_train, y_pred_ridge_train):.3f}")
print(f"Test R²:  {r2_score(y_test, y_pred_ridge_test):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge_test)):.1f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_ridge_test):.1f}")

# ============================================
# MODEL 3: RANDOM FOREST
# ============================================

print("\n" + "-"*60)
print("Model 3: Random Forest Regressor")
print("-"*60)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf_train = rf.predict(X_train)
y_pred_rf_test = rf.predict(X_test)

print(f"Train R²: {r2_score(y_train, y_pred_rf_train):.3f}")
print(f"Test R²:  {r2_score(y_test, y_pred_rf_test):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf_test)):.1f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_rf_test):.1f}")

# feature importance
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================
# MODEL 4: GRADIENT BOOSTING
# ============================================

print("\n" + "-"*60)
print("Model 4: Gradient Boosting Regressor")
print("-"*60)

gbr = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gbr.fit(X_train, y_train)

y_pred_gbr_train = gbr.predict(X_train)
y_pred_gbr_test = gbr.predict(X_test)

print(f"Train R²: {r2_score(y_train, y_pred_gbr_train):.3f}")
print(f"Test R²:  {r2_score(y_test, y_pred_gbr_test):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_gbr_test)):.1f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_gbr_test):.1f}")

# ============================================
# MODEL 5: XGBOOST
# ============================================

print("\n" + "-"*60)
print("Model 5: XGBoost Regressor")
print("-"*60)

xgb = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

y_pred_xgb_train = xgb.predict(X_train)
y_pred_xgb_test = xgb.predict(X_test)

print(f"Train R²: {r2_score(y_train, y_pred_xgb_train):.3f}")
print(f"Test R²:  {r2_score(y_test, y_pred_xgb_test):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb_test)):.1f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_xgb_test):.1f}")

# ============================================
# REGRESSION MODEL COMPARISON
# ============================================

print("\n" + "="*60)
print("REGRESSION MODEL COMPARISON")
print("="*60)

regression_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'Train_R2': [
        r2_score(y_train, y_pred_lr_train),
        r2_score(y_train, y_pred_ridge_train),
        r2_score(y_train, y_pred_rf_train),
        r2_score(y_train, y_pred_gbr_train),
        r2_score(y_train, y_pred_xgb_train)
    ],
    'Test_R2': [
        r2_score(y_test, y_pred_lr_test),
        r2_score(y_test, y_pred_ridge_test),
        r2_score(y_test, y_pred_rf_test),
        r2_score(y_test, y_pred_gbr_test),
        r2_score(y_test, y_pred_xgb_test)
    ],
    'Test_RMSE': [
        np.sqrt(mean_squared_error(y_test, y_pred_lr_test)),
        np.sqrt(mean_squared_error(y_test, y_pred_ridge_test)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf_test)),
        np.sqrt(mean_squared_error(y_test, y_pred_gbr_test)),
        np.sqrt(mean_squared_error(y_test, y_pred_xgb_test))
    ],
    'Test_MAE': [
        mean_absolute_error(y_test, y_pred_lr_test),
        mean_absolute_error(y_test, y_pred_ridge_test),
        mean_absolute_error(y_test, y_pred_rf_test),
        mean_absolute_error(y_test, y_pred_gbr_test),
        mean_absolute_error(y_test, y_pred_xgb_test)
    ]
})

print(regression_results.to_string(index=False))

# save results
regression_results.to_csv('output/regression_model_comparison.csv', index=False)

# identify best model
best_model_idx = regression_results['Test_R2'].idxmax()
best_model_name = regression_results.loc[best_model_idx, 'Model']
best_r2 = regression_results.loc[best_model_idx, 'Test_R2']

print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.3f})")

# ============================================
# TASK 2: CLASSIFICATION - PREDICT OUTBREAKS
# ============================================

print("\n" + "="*60)
print("TASK 2: CLASSIFICATION - PREDICTING DENGUE OUTBREAKS")
print("="*60)

# prepare data for classification
y_class = df_model['is_outbreak']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

print(f"\nClass distribution:")
print(f"Training: {y_train_c.value_counts().to_dict()}")
print(f"Testing: {y_test_c.value_counts().to_dict()}")

# ============================================
# CLASSIFICATION MODEL 1: RANDOM FOREST
# ============================================

print("\n" + "-"*60)
print("Classification Model 1: Random Forest")
print("-"*60)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train_c, y_train_c)

y_pred_rf_clf = rf_clf.predict(X_test_c)
y_pred_rf_clf_proba = rf_clf.predict_proba(X_test_c)[:, 1]

print(f"\nClassification Report:")
print(classification_report(y_test_c, y_pred_rf_clf))
print(f"ROC AUC Score: {roc_auc_score(y_test_c, y_pred_rf_clf_proba):.3f}")

# ============================================
# CLASSIFICATION MODEL 2: XGBOOST
# ============================================

print("\n" + "-"*60)
print("Classification Model 2: XGBoost")
print("-"*60)

xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(X_train_c, y_train_c)

y_pred_xgb_clf = xgb_clf.predict(X_test_c)
y_pred_xgb_clf_proba = xgb_clf.predict_proba(X_test_c)[:, 1]

print(f"\nClassification Report:")
print(classification_report(y_test_c, y_pred_xgb_clf))
print(f"ROC AUC Score: {roc_auc_score(y_test_c, y_pred_xgb_clf_proba):.3f}")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# 1. Feature Importance (Random Forest)
fig, ax = plt.subplots(figsize=(10, 6))
top_features = feature_importance.head(15)
ax.barh(top_features['feature'], top_features['importance'])
ax.set_xlabel('Importance', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('output/ml_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Actual vs Predicted (Best Regression Model)
best_predictions = {
    'Linear Regression': y_pred_lr_test,
    'Ridge': y_pred_ridge_test,
    'Random Forest': y_pred_rf_test,
    'Gradient Boosting': y_pred_gbr_test,
    'XGBoost': y_pred_xgb_test
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, y_pred) in enumerate(best_predictions.items()):
    ax = axes[idx]
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    
    # perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel('Actual Cases', fontsize=10)
    ax.set_ylabel('Predicted Cases', fontsize=10)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# hide last subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('output/ml_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Confusion Matrix (Classification)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest
cm_rf = confusion_matrix(y_test_c, y_pred_rf_clf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Random Forest Confusion Matrix')

# XGBoost
cm_xgb = confusion_matrix(y_test_c, y_pred_xgb_clf)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('XGBoost Confusion Matrix')

plt.tight_layout()
plt.savefig('output/ml_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(regression_results))
width = 0.35

ax.bar(x - width/2, regression_results['Train_R2'], width, label='Train R²', alpha=0.8)
ax.bar(x + width/2, regression_results['Test_R2'], width, label='Test R²', alpha=0.8)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regression_results['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/ml_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Machine learning pipeline complete!")
print("✓ All results saved to 'output/' folder")