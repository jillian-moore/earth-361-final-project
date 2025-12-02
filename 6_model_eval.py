# model evaluation and comparison

# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# load results ----
results_files = {
    'Boosted Tree Regression': 'results/bt_reg_results_table.csv',
    'Random Forest Regression': 'results/rf_reg_results_table.csv',
    'Boosted Tree Classification': 'results/bt_class_results_table.csv',
    'Random Forest Classification': 'results/rf_class_results_table.csv'
}

results = {}
for name, file in results_files.items():
    try:
        df = pd.read_csv(file)
        results[name] = df.iloc[0].to_dict()  # each CSV is a single-row dataframe
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping {name}.")

# regression comparison ----
regression_models = {k: v for k, v in results.items() if 'Regression' in k}

if regression_models:
    reg_comparison = pd.DataFrame({
        'Model': list(regression_models.keys()),
        'Test RMSE': [v['test_rmse'] for v in regression_models.values()],
        'Test MAE': [v['test_mae'] for v in regression_models.values()],
        'Test R²': [v['test_r2'] for v in regression_models.values()],
        'CV RMSE': [v['cv_rmse'] for v in regression_models.values()]
    })
    
    print("\nRegression Model Performance:")
    print(reg_comparison.to_string(index=False))
    
    # best regression model (lowest Test RMSE)
    best_reg_idx = reg_comparison['Test RMSE'].idxmin()
    best_reg_model = reg_comparison.loc[best_reg_idx, 'Model']
    print(f"\n BEST REGRESSION MODEL: {best_reg_model}")
    print(f"   Test RMSE: {reg_comparison.loc[best_reg_idx, 'Test RMSE']:.2f}")
    print(f"   Test R²: {reg_comparison.loc[best_reg_idx, 'Test R²']:.4f}")

# classification comparison ----
classification_models = {k: v for k, v in results.items() if v['task'] == 'classification'}

if classification_models:
    # only metrics both models have
    common_metrics = ['test_accuracy', 'test_f1']
    
    class_comparison = pd.DataFrame({
        'Model': list(classification_models.keys()),
        **{metric: [v[metric] for v in classification_models.values()] for metric in common_metrics}
    })
    
    print("\nClassification Model Performance:")
    print(class_comparison.to_string(index=False))
    
    # best model by F1
    best_class_idx = class_comparison['test_f1'].idxmax()
    best_class_model = class_comparison.loc[best_class_idx, 'Model']
    print(f"\n BEST CLASSIFICATION MODEL: {best_class_model}")
    print(f"   Test F1 Score: {class_comparison.loc[best_class_idx, 'test_f1']:.4f}")
    print(f"   Test Accuracy: {class_comparison.loc[best_class_idx, 'test_accuracy']:.4f}")