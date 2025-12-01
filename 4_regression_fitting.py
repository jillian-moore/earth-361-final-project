# regression analysis and modeling

# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

print(f"Data shape: {fever_df.shape}")
print(f"Variables: {fever_df.columns.tolist()}")

# prepare features ----
# select key predictors
features = ['temp_avg', 'precipitation_avg', 'humidity_avg', 'elevation']
target = 'cases'

# remove missing values
model_data = fever_df[features + [target]].dropna()

print(f"\nModel data shape: {model_data.shape}")
print(f"Cases range: {model_data[target].min():.0f} to {model_data[target].max():.0f}")

# split train/test ----
X = model_data[features]
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================
# MODEL 1: SIMPLE LINEAR REGRESSION
# ============================================

print("\n" + "="*60)
print("MODEL 1: SIMPLE LINEAR REGRESSION")
print("="*60)

# fit model
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

# predictions
y_pred_train_linear = model_linear.predict(X_train)
y_pred_test_linear = model_linear.predict(X_test)

# evaluate
r2_train_linear = r2_score(y_train, y_pred_train_linear)
r2_test_linear = r2_score(y_test, y_pred_test_linear)
rmse_train_linear = np.sqrt(mean_squared_error(y_train, y_pred_train_linear))
rmse_test_linear = np.sqrt(mean_squared_error(y_test, y_pred_test_linear))
mae_test_linear = mean_absolute_error(y_test, y_pred_test_linear)

print(f"\nCoefficients:")
for feature, coef in zip(features, model_linear.coef_):
    print(f"  {feature:20s}: {coef:8.2f}")
print(f"  {'Intercept':20s}: {model_linear.intercept_:8.2f}")

print(f"\nPerformance:")
print(f"  Train R²: {r2_train_linear:.3f}")
print(f"  Test R²:  {r2_test_linear:.3f}")
print(f"  Test RMSE: {rmse_test_linear:.1f}")
print(f"  Test MAE:  {mae_test_linear:.1f}")

# ============================================
# MODEL 2: INTERACTION REGRESSION
# ============================================

print("\n" + "="*60)
print("MODEL 2: INTERACTION REGRESSION")
print("="*60)

# create interaction terms
X_train_interact = X_train.copy()
X_test_interact = X_test.copy()

# temp × precipitation
X_train_interact['temp_precip'] = X_train['temp_avg'] * X_train['precipitation_avg']
X_test_interact['temp_precip'] = X_test['temp_avg'] * X_test['precipitation_avg']

# temp × humidity
X_train_interact['temp_humid'] = X_train['temp_avg'] * X_train['humidity_avg']
X_test_interact['temp_humid'] = X_test['temp_avg'] * X_test['humidity_avg']

# precipitation × humidity
X_train_interact['precip_humid'] = X_train['precipitation_avg'] * X_train['humidity_avg']
X_test_interact['precip_humid'] = X_test['precipitation_avg'] * X_test['humidity_avg']

# fit model
model_interact = LinearRegression()
model_interact.fit(X_train_interact, y_train)

# predictions
y_pred_train_interact = model_interact.predict(X_train_interact)
y_pred_test_interact = model_interact.predict(X_test_interact)

# evaluate
r2_train_interact = r2_score(y_train, y_pred_train_interact)
r2_test_interact = r2_score(y_test, y_pred_test_interact)
rmse_train_interact = np.sqrt(mean_squared_error(y_train, y_pred_train_interact))
rmse_test_interact = np.sqrt(mean_squared_error(y_test, y_pred_test_interact))
mae_test_interact = mean_absolute_error(y_test, y_pred_test_interact)

print(f"\nPerformance:")
print(f"  Train R²: {r2_train_interact:.3f}")
print(f"  Test R²:  {r2_test_interact:.3f}")
print(f"  Test RMSE: {rmse_test_interact:.1f}")
print(f"  Test MAE:  {mae_test_interact:.1f}")

# ============================================
# MODEL 3: LOG-TRANSFORMED REGRESSION
# ============================================

print("\n" + "="*60)
print("MODEL 3: LOG-TRANSFORMED REGRESSION")
print("="*60)

# log transform target (handle zeros with log1p)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# fit model
model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

# predictions (transform back to original scale)
y_pred_train_log_scale = model_log.predict(X_train)
y_pred_test_log_scale = model_log.predict(X_test)

y_pred_train_log = np.expm1(y_pred_train_log_scale)
y_pred_test_log = np.expm1(y_pred_test_log_scale)

# evaluate
r2_train_log = r2_score(y_train, y_pred_train_log)
r2_test_log = r2_score(y_test, y_pred_test_log)
rmse_train_log = np.sqrt(mean_squared_error(y_train, y_pred_train_log))
rmse_test_log = np.sqrt(mean_squared_error(y_test, y_pred_test_log))
mae_test_log = mean_absolute_error(y_test, y_pred_test_log)

print(f"\nPerformance:")
print(f"  Train R²: {r2_train_log:.3f}")
print(f"  Test R²:  {r2_test_log:.3f}")
print(f"  Test RMSE: {rmse_test_log:.1f}")
print(f"  Test MAE:  {mae_test_log:.1f}")

# ============================================
# MODEL 4: POLYNOMIAL (NONLINEAR) REGRESSION
# ============================================

print("\n" + "="*60)
print("MODEL 4: POLYNOMIAL REGRESSION (degree=2)")
print("="*60)

# create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f"\nPolynomial features: {X_train_poly.shape[1]} (from {X_train.shape[1]} original)")

# standardize features for better convergence
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# fit model
model_poly = LinearRegression()
model_poly.fit(X_train_poly_scaled, y_train)

# predictions
y_pred_train_poly = model_poly.predict(X_train_poly_scaled)
y_pred_test_poly = model_poly.predict(X_test_poly_scaled)

# evaluate
r2_train_poly = r2_score(y_train, y_pred_train_poly)
r2_test_poly = r2_score(y_test, y_pred_test_poly)
rmse_train_poly = np.sqrt(mean_squared_error(y_train, y_pred_train_poly))
rmse_test_poly = np.sqrt(mean_squared_error(y_test, y_pred_test_poly))
mae_test_poly = mean_absolute_error(y_test, y_pred_test_poly)

print(f"\nPerformance:")
print(f"  Train R²: {r2_train_poly:.3f}")
print(f"  Test R²:  {r2_test_poly:.3f}")
print(f"  Test RMSE: {rmse_test_poly:.1f}")
print(f"  Test MAE:  {mae_test_poly:.1f}")

# ============================================
# MODEL COMPARISON
# ============================================

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

results = pd.DataFrame({
    'Model': ['Linear', 'Interaction', 'Log-Transform', 'Polynomial'],
    'Train_R2': [r2_train_linear, r2_train_interact, r2_train_log, r2_train_poly],
    'Test_R2': [r2_test_linear, r2_test_interact, r2_test_log, r2_test_poly],
    'Test_RMSE': [rmse_test_linear, rmse_test_interact, rmse_test_log, rmse_test_poly],
    'Test_MAE': [mae_test_linear, mae_test_interact, mae_test_log, mae_test_poly]
})

print(results.to_string(index=False))

# save results
results.to_csv('output/model_comparison.csv', index=False)

# ============================================
# VISUALIZATION: PREDICTED VS ACTUAL
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

models_pred = [
    ('Linear', y_pred_test_linear),
    ('Interaction', y_pred_test_interact),
    ('Log-Transform', y_pred_test_log),
    ('Polynomial', y_pred_test_poly)
]

for ax, (name, y_pred) in zip(axes.flat, models_pred):
    # scatter plot
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    
    # perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    # labels
    ax.set_xlabel('Actual Cases', fontsize=10)
    ax.set_ylabel('Predicted Cases', fontsize=10)
    ax.set_title(f'{name} Model', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # add R² to plot
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('output/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# VISUALIZATION: 3D SURFACE PLOT
# ============================================

print("\n" + "="*60)
print("CREATING 3D SURFACE PLOT")
print("="*60)

# create grid for temperature and precipitation
temp_range = np.linspace(X['temp_avg'].min(), X['temp_avg'].max(), 50)
precip_range = np.linspace(X['precipitation_avg'].min(), X['precipitation_avg'].max(), 50)

temp_grid, precip_grid = np.meshgrid(temp_range, precip_range)

# use median values for other features
humidity_median = X['humidity_avg'].median()
elevation_median = X['elevation'].median()

# create prediction grid using best model (polynomial)
grid_points = pd.DataFrame({
    'temp_avg': temp_grid.ravel(),
    'precipitation_avg': precip_grid.ravel(),
    'humidity_avg': humidity_median,
    'elevation': elevation_median
})

# transform and predict
grid_poly = poly.transform(grid_points[features])
grid_poly_scaled = scaler.transform(grid_poly)
cases_pred = model_poly.predict(grid_poly_scaled)
cases_grid = cases_pred.reshape(temp_grid.shape)

# 3D surface plot ----
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(temp_grid, precip_grid, cases_grid,
                       cmap='viridis', alpha=0.8,
                       edgecolor='none', antialiased=True)

# add actual data points
ax.scatter(X_test['temp_avg'], X_test['precipitation_avg'], y_test,
           c='red', marker='o', s=20, alpha=0.6, label='Actual Data')

# labels
ax.set_xlabel('Temperature (°C)', fontsize=11, labelpad=10)
ax.set_ylabel('Precipitation (mm)', fontsize=11, labelpad=10)
ax.set_zlabel('Dengue Cases', fontsize=11, labelpad=10)
ax.set_title('Dengue Cases as Function of Temperature and Precipitation\n(Polynomial Model)',
             fontsize=12, fontweight='bold', pad=20)

# colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Predicted Cases')

# legend
ax.legend(loc='upper left', fontsize=9)

# viewing angle
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('output/3d_surface_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# VISUALIZATION: 2D CONTOUR PLOT
# ============================================

fig, ax = plt.subplots(figsize=(10, 8))

# contour plot
contour = ax.contourf(temp_grid, precip_grid, cases_grid, 
                      levels=20, cmap='viridis', alpha=0.8)

# add actual data points
scatter = ax.scatter(X_test['temp_avg'], X_test['precipitation_avg'],
                    c=y_test, cmap='Reds', s=30, edgecolors='black',
                    linewidth=0.5, alpha=0.7, label='Actual Data')

# labels
ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('Precipitation (mm)', fontsize=11)
ax.set_title('Dengue Cases: Temperature × Precipitation (Contour Plot)',
             fontsize=12, fontweight='bold')

# colorbars
cbar1 = plt.colorbar(contour, ax=ax, label='Predicted Cases (Model)')
cbar2 = plt.colorbar(scatter, ax=ax, label='Actual Cases')

ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('output/2d_contour_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# FEATURE IMPORTANCE (from linear model)
# ============================================

fig, ax = plt.subplots(figsize=(8, 5))

feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model_linear.coef_
})

feature_importance = feature_importance.sort_values('Coefficient', ascending=True)

colors = ['coral' if x < 0 else 'steelblue' for x in feature_importance['Coefficient']]
ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient Value', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.set_title('Feature Importance (Linear Model Coefficients)', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ All models trained and visualizations saved to 'output/' folder")