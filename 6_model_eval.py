import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load all results
results_files = {
    'XGBoost Regression': 'xgb_regression_results.json',
    'Random Forest Regression': 'rf_regression_results.json',
    'Linear Regression': 'linear_regression_results.json',
    'XGBoost Classification': 'xgb_classification_results.json',
    'Random Forest Classification': 'rf_classification_results.json'
}

results = {}
for name, file in results_files.items():
    try:
        with open(file, 'r') as f:
            results[name] = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping {name}.")

# ===== REGRESSION COMPARISON =====
print("="*60)
print("REGRESSION MODELS COMPARISON")
print("="*60)

regression_models = {k: v for k, v in results.items() if v['task'] == 'regression'}

if regression_models:
    # Create comparison dataframe
    reg_comparison = pd.DataFrame({
        'Model': list(regression_models.keys()),
        'Test RMSE': [v['test_rmse'] for v in regression_models.values()],
        'Test MAE': [v['test_mae'] for v in regression_models.values()],
        'Test R²': [v['test_r2'] for v in regression_models.values()],
        'CV RMSE': [v['cv_rmse'] for v in regression_models.values()]
    })
    
    print("\nRegression Model Performance:")
    print(reg_comparison.to_string(index=False))
    
    # Find best regression model
    best_reg_idx = reg_comparison['Test RMSE'].idxmin()
    best_reg_model = reg_comparison.loc[best_reg_idx, 'Model']
    print(f"\n🏆 BEST REGRESSION MODEL: {best_reg_model}")
    print(f"   Test RMSE: {reg_comparison.loc[best_reg_idx, 'Test RMSE']:.2f}")
    print(f"   Test R²: {reg_comparison.loc[best_reg_idx, 'Test R²']:.4f}")

# ===== CLASSIFICATION COMPARISON =====
print("\n" + "="*60)
print("CLASSIFICATION MODELS COMPARISON")
print("="*60)

classification_models = {k: v for k, v in results.items() if v['task'] == 'classification'}

if classification_models:
    # Create comparison dataframe
    class_comparison = pd.DataFrame({
        'Model': list(classification_models.keys()),
        'Test Accuracy': [v['test_accuracy'] for v in classification_models.values()],
        'Precision': [v['precision'] for v in classification_models.values()],
        'Recall': [v['recall'] for v in classification_models.values()],
        'F1 Score': [v['f1_score'] for v in classification_models.values()],
        'ROC AUC': [v['roc_auc'] for v in classification_models.values()],
        'CV F1': [v['cv_f1'] for v in classification_models.values()]
    })
    
    print("\nClassification Model Performance:")
    print(class_comparison.to_string(index=False))
    
    # Find best classification model
    best_class_idx = class_comparison['F1 Score'].idxmax()
    best_class_model = class_comparison.loc[best_class_idx, 'Model']
    print(f"\n🏆 BEST CLASSIFICATION MODEL: {best_class_model}")
    print(f"   F1 Score: {class_comparison.loc[best_class_idx, 'F1 Score']:.4f}")
    print(f"   ROC AUC: {class_comparison.loc[best_class_idx, 'ROC AUC']:.4f}")

# ===== VISUALIZATIONS =====
fig = plt.figure(figsize=(18, 12))

# --- REGRESSION PLOTS ---
if regression_models:
    # 1. RMSE Comparison
    ax1 = plt.subplot(3, 3, 1)
    x = np.arange(len(reg_comparison))
    width = 0.35
    ax1.bar(x - width/2, reg_comparison['Test RMSE'], width, label='Test RMSE', alpha=0.8)
    ax1.bar(x + width/2, reg_comparison['CV RMSE'], width, label='CV RMSE', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Regression: RMSE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_comparison['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. R² Comparison
    ax2 = plt.subplot(3, 3, 2)
    colors = ['#2ecc71' if i == best_reg_idx else '#3498db' for i in range(len(reg_comparison))]
    ax2.bar(reg_comparison['Model'], reg_comparison['Test R²'], color=colors, alpha=0.8)
    ax2.set_ylabel('R² Score')
    ax2.set_title('Regression: R² Score Comparison')
    ax2.set_xticklabels(reg_comparison['Model'], rotation=45, ha='right')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # 3. Best Regression Model - Actual vs Predicted
    ax3 = plt.subplot(3, 3, 3)
    best_reg_pred_file = best_reg_model.lower().replace(' ', '_') + '_predictions.csv'
    try:
        best_reg_pred = pd.read_csv(best_reg_pred_file)
        ax3.scatter(best_reg_pred['actual'], best_reg_pred['predicted'], alpha=0.5, s=30)
        min_val = min(best_reg_pred['actual'].min(), best_reg_pred['predicted'].min())
        max_val = max(best_reg_pred['actual'].max(), best_reg_pred['predicted'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual Cases')
        ax3.set_ylabel('Predicted Cases')
        ax3.set_title(f'{best_reg_model}\nActual vs Predicted')
        ax3.legend()
        ax3.grid(alpha=0.3)
    except FileNotFoundError:
        ax3.text(0.5, 0.5, 'Predictions file not found', ha='center', va='center')

# --- CLASSIFICATION PLOTS ---
if classification_models:
    # 4. F1 Score Comparison
    ax4 = plt.subplot(3, 3, 4)
    colors = ['#2ecc71' if i == best_class_idx else '#3498db' for i in range(len(class_comparison))]
    ax4.bar(class_comparison['Model'], class_comparison['F1 Score'], color=colors, alpha=0.8)
    ax4.set_ylabel('F1 Score')
    ax4.set_title('Classification: F1 Score Comparison')
    ax4.set_xticklabels(class_comparison['Model'], rotation=45, ha='right')
    ax4.grid(alpha=0.3)
    
    # 5. ROC AUC Comparison
    ax5 = plt.subplot(3, 3, 5)
    ax5.bar(class_comparison['Model'], class_comparison['ROC AUC'], color=colors, alpha=0.8)
    ax5.set_ylabel('ROC AUC')
    ax5.set_title('Classification: ROC AUC Comparison')
    ax5.set_xticklabels(class_comparison['Model'], rotation=45, ha='right')
    ax5.grid(alpha=0.3)
    ax5.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
    ax5.legend()
    
    # 6. Best Classification Model - Confusion Matrix
    ax6 = plt.subplot(3, 3, 6)
    best_class_cm = np.array(results[best_class_model]['confusion_matrix'])
    sns.heatmap(best_class_cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=['No Outbreak', 'Outbreak'],
                yticklabels=['No Outbreak', 'Outbreak'])
    ax6.set_ylabel('Actual')
    ax6.set_xlabel('Predicted')
    ax6.set_title(f'{best_class_model}\nConfusion Matrix')

# --- FEATURE IMPORTANCE PLOTS ---
# 7. Top 10 features - Best Regression Model
if regression_models:
    ax7 = plt.subplot(3, 3, 7)
    feat_imp = results[best_reg_model]['feature_importance']
    top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    features, importances = zip(*top_features)
    ax7.barh(range(len(features)), importances, alpha=0.8)
    ax7.set_yticks(range(len(features)))
    ax7.set_yticklabels(features)
    ax7.set_xlabel('Importance')
    ax7.set_title(f'{best_reg_model}\nTop 10 Features')
    ax7.invert_yaxis()
    ax7.grid(alpha=0.3)

# 8. Top 10 features - Best Classification Model
if classification_models:
    ax8 = plt.subplot(3, 3, 8)
    feat_imp = results[best_class_model]['feature_importance']
    top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    features, importances = zip(*top_features)
    ax8.barh(range(len(features)), importances, alpha=0.8)
    ax8.set_yticks(range(len(features)))
    ax8.set_yticklabels(features)
    ax8.set_xlabel('Importance')
    ax8.set_title(f'{best_class_model}\nTop 10 Features')
    ax8.invert_yaxis()
    ax8.grid(alpha=0.3)

# 9. Precision-Recall-F1 Comparison
if classification_models:
    ax9 = plt.subplot(3, 3, 9)
    x = np.arange(len(class_comparison))
    width = 0.25
    ax9.bar(x - width, class_comparison['Precision'], width, label='Precision', alpha=0.8)
    ax9.bar(x, class_comparison['Recall'], width, label='Recall', alpha=0.8)
    ax9.bar(x + width, class_comparison['F1 Score'], width, label='F1 Score', alpha=0.8)
    ax9.set_xlabel('Model')
    ax9.set_ylabel('Score')
    ax9.set_title('Classification: Precision-Recall-F1')
    ax9.set_xticks(x)
    ax9.set_xticklabels(class_comparison['Model'], rotation=45, ha='right')
    ax9.legend()
    ax9.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Visualization saved as 'model_comparison_results.png'")

plt.show()

# Save summary
summary = {
    'best_regression_model': best_reg_model if regression_models else None,
    'best_regression_metrics': reg_comparison.loc[best_reg_idx].to_dict() if regression_models else None,
    'best_classification_model': best_class_model if classification_models else None,
    'best_classification_metrics': class_comparison.loc[best_class_idx].to_dict() if classification_models else None
}

with open('model_comparison_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("\n✅ Analysis complete! Summary saved to 'model_comparison_summary.json'")