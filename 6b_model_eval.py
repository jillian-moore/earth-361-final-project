# regression model analysis

# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load results ----
results_files = {
    'Simple Linear Regression': 'results/simple_reg_results_table.csv',
    'Interaction Terms Regression': 'results/interaction_reg_results_table.csv',
    'Autoregressive Regression': 'results/auto_reg_results_table.csv',
    'Lagged 12-Week Regression': 'results/lag_reg_results_table.csv'
}

# compile results into dataframes ----
all_results_table = []
all_results_viz = []

for model_name, file_path in results_files.items():
    try:
        df = pd.read_csv(file_path)
        
        result_table = {
            'Model': model_name,
            'Test R²': df['test_r2'].iloc[0],
            'Test RMSE': df['test_rmse'].iloc[0],
            'Test MAE': df['test_mae'].iloc[0]
        }
        all_results_table.append(result_table)
        
        result_viz = {
            'Model': model_name,
            'Train R²': df['train_r2'].iloc[0],
            'Test R²': df['test_r2'].iloc[0],
            'Train RMSE': df['train_rmse'].iloc[0],
            'Test RMSE': df['test_rmse'].iloc[0],
            'Train MAE': df['train_mae'].iloc[0],
            'Test MAE': df['test_mae'].iloc[0]
        }
        all_results_viz.append(result_viz)
        
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, skipping {model_name}")

comparison_df_table = pd.DataFrame(all_results_table)
comparison_df_viz = pd.DataFrame(all_results_viz)

# sort by test R² (descending) ----
comparison_df_table = comparison_df_table.sort_values('Test R²', ascending=False)
comparison_df_viz = comparison_df_viz.sort_values('Test R²', ascending=False)

# save comparison table ----
comparison_df_table.to_csv('results/regression_comparison.csv', index=False)

# create latex table ----
latex_table = comparison_df_table.to_latex(
    index=False,
    float_format="%.3f",
    column_format='lccc',
    escape=False
)
with open('results/reg_model_comparison.tex', 'w') as f:
    f.write(latex_table)

# create visualization ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# create shorter model names for display
model_names_short = []
for model in comparison_df_viz['Model']:
    if 'Simple Linear' in model:
        model_names_short.append('Simple Linear')
    elif 'Interaction' in model:
        model_names_short.append('Interaction Terms')
    elif 'Autoregressive' in model:
        model_names_short.append('Autoregressive')
    elif 'Lagged' in model:
        model_names_short.append('Lagged 12-Week')
    else:
        model_names_short.append(model)

# plot 1: R² comparison
ax1 = axes[0]
x = np.arange(len(comparison_df_viz))
width = 0.35
ax1.barh(x - width/2, comparison_df_viz['Train R²'], width, label='Train', alpha=0.8, color='steelblue')
ax1.barh(x + width/2, comparison_df_viz['Test R²'], width, label='Test', alpha=0.8, color='coral')
ax1.set_yticks(x)
ax1.set_yticklabels(model_names_short, fontsize=9)
ax1.set_xlabel('R² Score', fontsize=10)
ax1.set_title('Model R² Comparison', fontsize=18)
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# plot 2: RMSE comparison
ax2 = axes[1]
ax2.barh(x - width/2, comparison_df_viz['Train RMSE'], width, label='Train', alpha=0.8, color='steelblue')
ax2.barh(x + width/2, comparison_df_viz['Test RMSE'], width, label='Test', alpha=0.8, color='coral')
ax2.set_yticks(x)
ax2.set_yticklabels(model_names_short, fontsize=9)
ax2.set_xlabel('RMSE (lower is better)', fontsize=10)
ax2.set_title('Model RMSE Comparison', fontsize=18)
ax2.legend(fontsize=9)
ax2.grid(axis='x', alpha=0.3)

# plot 3: MAE comparison
ax3 = axes[2]
ax3.barh(x - width/2, comparison_df_viz['Train MAE'], width, label='Train', alpha=0.8, color='steelblue')
ax3.barh(x + width/2, comparison_df_viz['Test MAE'], width, label='Test', alpha=0.8, color='coral')
ax3.set_yticks(x)
ax3.set_yticklabels(model_names_short, fontsize=9)
ax3.set_xlabel('MAE (lower is better)', fontsize=10)
ax3.set_title('Model MAE Comparison', fontsize=18)
ax3.legend(fontsize=9)
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/regression_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/regression_model_comparison.pdf', bbox_inches='tight')  # PDF for LaTeX
plt.show()