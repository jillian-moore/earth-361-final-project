
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Sample data based on the image (you'll need to expand this with your full dataset)
df = pd.read_csv('data/raw/Dengue Hotspot Data.csv')

print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# Prepare features and target
# Assuming 'total_cases' is your target variable - adjust if different
target_col = 'total_cases'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Handle any missing values
X = X.fillna(X.mean())
y = y.fillna(0)

print(f"\nTarget variable distribution:")
print(y.describe())

if len(df) > 10:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"\nTest Set Metrics:")
    print(f"  Mean Absolute Error: {mae:.2f} cases")
    print(f"  Root Mean Squared Error: {rmse:.2f} cases")
    print(f"  R² Score: {r2:.3f}")
    
    print(f"\nTraining Set Metrics:")
    print(f"  Mean Absolute Error: {train_mae:.2f} cases")
    print(f"  R² Score: {train_r2:.3f}")
    
    if r2 > 0.7:
        print("\n✅ Model shows GOOD predictive accuracy (R² > 0.7)")
    elif r2 > 0.4:
        print("\n⚠️  Model shows MODERATE predictive accuracy (R² > 0.4)")
    else:
        print("\n❌ Model shows LOW predictive accuracy (R² < 0.4)")
        print("   Consider: feature engineering, more data, or different model")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*50)
    print(feature_importance.head(10).to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Cases')
    plt.ylabel('Predicted Cases')
    plt.title('Actual vs Predicted')
    
    plt.subplot(1, 2, 2)
    plt.barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
else:
    print("⚠️ Warning: Need more data points for proper model training")
    print(f"Current dataset size: {len(df)} rows")
    print("\nTo test this properly:")
    print("1. Load your full dengue dataset")
    print("2. Ensure you have at least 100+ rows")
    print("3. Re-run the model training")
    
    # Simple baseline with limited data
    print(f"\nCurrent data summary:")
    print(df.describe())




    