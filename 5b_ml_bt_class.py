# boosted tree classification

# load packages ----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# load data ----
df = pd.read_csv('data/processed/dengue_data_cleaned.csv')

# define features (exclude target and non-predictive columns) ----
exclude_cols = ['total_cases', 'date', 'is_outbreak', 'log_cases', 
                'cases_zscore', 'cases_lag', 'cases_3mo_avg']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# prepare data ----
X = df[feature_cols]
y = df['is_outbreak']

# encode categorical variables ----
categorical_cols = ['city', 'season', 'temp_category', 'rain_category', 'humidity_category']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# split data ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# define repeated stratified k-fold cross-validation ----
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# define hyperparameter grid ----
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5],
    'reg_lambda': [1, 5, 10] # L2 regularization
}

# initialize base model ----
base_model = xgb.XGBClassifier(
    random_state=42,
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss'
)

# perform grid search with cross-validation ----
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=rskf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# get best model ----
model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

# predictions ----
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_prob_test = model.predict_proba(X_test)[:, 1]

# evaluate ----
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, y_pred_prob_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)

# results ----
results = {
    'model_type': 'XGBoost Classification',
    'task': 'classification',
    'best_params': best_params,
    'cv_config': {'n_splits': 5, 'n_repeats': 3},
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'test_f1': float(test_f1),
    'test_auc': float(test_auc),
    'confusion_matrix': conf_matrix.tolist(),
    'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist()))
}

print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test Precision: {test_precision:.3f}")
print(f"Test Recall: {test_recall:.3f}")
print(f"Test F1: {test_f1:.3f}")
print(f"Test ROC AUC: {test_auc:.3f}")

# save model and results ----
results_df = pd.DataFrame([results])
results_df.to_csv("results/bt_class_results_table.csv", index=False)
results_df.to_latex("results/bt_class_results_table.tex", index=False, float_format="%.3f")

# save predictions ----
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test,
    'pred_prob': y_pred_prob_test
})
predictions_df.to_csv('results/bt_class_predictions.csv', index=False)

# visualizations ----
# confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/bt_class_confusion_matrix.png", dpi=300)
plt.show()

# feature importance plot
xgb.plot_importance(model, max_num_features=20, height=0.5)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig("results/bt_class_feature_importance.png", dpi=300)
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/bt_class_roc_curve.png", dpi=300)
plt.show()