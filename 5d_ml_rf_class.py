# random forest classification

# load packages ----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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
y = df['is_outbreak']  # target is now classification

# encode categorical variables ----
categorical_cols = ['city', 'season', 'temp_category', 'rain_category', 'humidity_category']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# split data ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# define repeated stratified k-fold cross-validation ----
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# define hyperparameter grid ----
param_grid = {
    'n_estimators': [100, 300, 500],          # number of trees
    'min_samples_leaf': [1, 3, 5],            # node size
    'max_features': ['sqrt', 'log2', 0.3, 0.5] # features sampled per split
}

# initialize base model ----
base_model = RandomForestClassifier(random_state=42)

# perform grid search with cross-validation ----
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=rskf,
    scoring='f1',  # optimize F1-score
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

# evaluate ----
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
train_f1 = f1_score(y_train, y_pred_train)
test_f1 = f1_score(y_test, y_pred_test)
train_precision = precision_score(y_train, y_pred_train)
test_precision = precision_score(y_test, y_pred_test)
train_recall = recall_score(y_train, y_pred_train)
test_recall = recall_score(y_test, y_pred_test)

# results ----
results = {
    'model_type': 'Random Forest Classification',
    'task': 'classification',
    'best_params': best_params,
    'cv_config': {'n_splits': 5, 'n_repeats': 3},
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'train_f1': float(train_f1),
    'test_f1': float(test_f1),
    'train_precision': float(train_precision),
    'test_precision': float(test_precision),
    'train_recall': float(train_recall),
    'test_recall': float(test_recall),
    'cv_f1': float(best_cv_score),
    'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist()))
}

print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test F1-score: {test_f1:.3f}")
print(f"CV F1-score: {best_cv_score:.3f}")

# save model and results ----
results_df = pd.DataFrame([results])
results_df.to_csv("results/rf_class_results_table.csv", index=False)
results_df.to_latex("results/rf_class_results_table.tex", index=False, float_format="%.3f")

# predictions ----
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test
})
predictions_df.to_csv('results/rf_classification_predictions.csv', index=False)

# visualizations ----
# confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Random Forest)')
plt.tight_layout()
plt.savefig("results/rf_class_confusion_matrix.png", dpi=300)
plt.show()

# feature importance plot
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
importances.head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("results/rf_class_feature_importance.png", dpi=300)
plt.show()
