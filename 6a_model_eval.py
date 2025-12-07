# model evaluation and comparison

# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# personal examinations ----
import pprint
sol= pd.read_csv("results/bt_reg_results_table.csv")
pprint.pprint(sol['feature_importance'].iloc[0])

# load results ----
results_files = {
    'Boosted Tree Regression': 'results/bt_reg_results_table.csv',
    'Random Forest Regression': 'results/rf_reg_results_table.csv',
    'Boosted Tree Classification': 'results/bt_class_results_table.csv',
    'Random Forest Classification': 'results/rf_class_results_table.csv',
    'Elastic Net Regression': 'results/elasticnet_results_table.csv'
}

results = {}
for name, file in results_files.items():
    try:
        df = pd.read_csv(file)
        results[name] = df.iloc[0].to_dict()
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping {name}.")

# regression comparison ----
regression_models = {k: v for k, v in results.items() if 'Regression' in k}

if regression_models:
    reg_comparison = pd.DataFrame({
        'Model': list(regression_models.keys()),
        'R²': [v['test_r2'] for v in regression_models.values()],
        'RMSE': [v['test_rmse'] for v in regression_models.values()],
        'MAE': [v['test_mae'] for v in regression_models.values()]
    })
    
    # sort by RMSE (lower is better)
    reg_comparison = reg_comparison.sort_values('RMSE')
    
    # save to csv
    reg_comparison.to_csv('results/regression_comparison.csv', index=False)
    # save to .tex
    reg_comparison.to_latex(
        'results/reg_comparison.tex',
        index=False,
        escape=False,          # allows LaTeX symbols like % or _
        caption=None,
        label=None
    )

    print(reg_comparison.to_string(index=False))

# classification comparison ----
classification_models = {k: v for k, v in results.items() if 'Classification' in k}

if classification_models:
    # check available columns
    if classification_models:
        first_model = list(classification_models.values())[0]
        print(f"Available classification columns: {list(first_model.keys())}\n")
    
    class_comparison = pd.DataFrame({
        'Model': list(classification_models.keys()),
        'Recall': [v.get('test_recall', v.get('recall', np.nan)) for v in classification_models.values()],
        'ROC AUC': [v.get('test_auc', v.get('roc_auc', np.nan)) for v in classification_models.values()]
    })
    
    # sort by ROC AUC (higher is better)
    class_comparison = class_comparison.sort_values('ROC AUC', ascending=False)
    
    # save to csv
    class_comparison.to_csv('results/classification_comparison.csv', index=False)
    # save to .tex
    class_comparison.to_latex(
        'results/classification_comparison.tex',
        index=False,
        escape=False,          # allows LaTeX symbols like % or _
        caption=None,
        label=None
    )

    print(class_comparison.to_string(index=False))