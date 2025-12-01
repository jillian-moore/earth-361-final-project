# numerical modeling: SIR model with climate-dependent transmission

# load packages ----
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# aggregate to national monthly totals ----
# data is 25 districts x 36 months = 900 rows
# need to sum districts to get 36 monthly national totals
df = fever_df.groupby(['year', 'month']).agg({
    'cases': 'sum',
    'temp_avg': 'mean',
    'precipitation_avg': 'mean'
}).reset_index()

df = df.sort_values(['year', 'month']).reset_index(drop=True)

print(f"Total months: {len(df)}")
print(f"Cases range: {df['cases'].min()} to {df['cases'].max()}")

# define parameters ----
N = 21982608  # Sri Lanka total population
gamma = (1/7) * 30  # recovery rate per month (7-day infection, 30-day month)

# calculate rate of change ----
df['dI_dt'] = df['cases'].diff().fillna(0)

# calculate S and R over time ----
# initialize
S_vals = np.zeros(len(df))
R_vals = np.zeros(len(df))

S_vals[0] = N - df.loc[0, 'cases']
R_vals[0] = 0

# update for each time step
for i in range(1, len(df)):
    # people recover at rate gamma per month
    R_vals[i] = R_vals[i-1] + gamma * df.loc[i-1, 'cases']
    # susceptible = total - infected - recovered
    S_vals[i] = N - df.loc[i, 'cases'] - R_vals[i]

df['S'] = S_vals
df['R'] = R_vals

# estimate empirical beta ----
# from SIR equation: dI/dt = β*S*I/N - γ*I
# rearranging: β = (dI/dt + γ*I) * N / (S*I)
beta_vals = []
for i, row in df.iterrows():
    I = row['cases']
    S = row['S']
    dI = row['dI_dt']
    
    if I > 0 and S > 0:
        beta = (dI + gamma * I) * N / (S * I)
        if beta > 0:
            beta_vals.append(beta)
        else:
            beta_vals.append(np.nan)
    else:
        beta_vals.append(np.nan)

df['beta'] = beta_vals

# fill missing beta values
df['beta'] = df['beta'].fillna(method='ffill').fillna(method='bfill')

print(f"\nBeta range: {df['beta'].min():.4f} to {df['beta'].max():.4f}")

# fit regression: β(T,R) ----
# model beta as function of temperature and rainfall
data = df[['temp_avg', 'precipitation_avg', 'beta']].dropna()

X_clean = data[['temp_avg', 'precipitation_avg']]
y_clean = data['beta']

model = LinearRegression().fit(X_clean, y_clean)

print(f"\nβ(T,R) = {model.intercept_:.4f} + {model.coef_[0]:.4f}*T + {model.coef_[1]:.4f}*R")

# define beta function ----
def beta_func(t_index):
    """Calculate transmission rate at time t based on climate"""
    T = df.loc[t_index, 'temp_avg']
    R = df.loc[t_index, 'precipitation_avg']
    beta = model.intercept_ + model.coef_[0]*T + model.coef_[1]*R
    return max(0.001, beta)  # ensure positive

# define SIR ODE system ----
def sir_ode(t, y):
    """SIR differential equations with time-varying beta"""
    S, I, R = y
    t_idx = int(np.clip(round(t), 0, len(df) - 1))
    beta = beta_func(t_idx)
    
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    
    return [dS, dI, dR]

# set initial conditions ----
I0 = df.loc[0, 'cases']
R0 = 0
S0 = N - I0 - R0

# solve SIR model ----
t_span = [0, len(df) - 1]
t_eval = np.arange(len(df))

sol = solve_ivp(sir_ode, t_span, [S0, I0, R0], t_eval=t_eval)

# extract predictions ----
S_pred = sol.y[0]
I_pred = sol.y[1]
R_pred = sol.y[2]

# evaluate model performance ----
r2 = r2_score(df['cases'], I_pred)
rmse = np.sqrt(mean_squared_error(df['cases'], I_pred))
mae = np.mean(np.abs(df['cases'] - I_pred))

# calculate R0 ----
mean_beta = df['beta'].mean()
R0 = mean_beta / gamma

# visualize results ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# plot 1: predictions vs observed
ax = axes[0]
ax.plot(df['cases'].values, 'o-', label='Observed Cases', 
        linewidth=2, markersize=6, color='steelblue')
ax.plot(I_pred, 's-', label='Predicted Cases', 
        linewidth=2, markersize=6, color='coral')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Number of Cases', fontsize=11)
ax.set_title('Observed vs Predicted Dengue Cases', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# plot 2: beta over time
ax = axes[1]
ax.plot(df['beta'].values, 'o-', label='Empirical β', 
        linewidth=2, color='forestgreen')
ax.axhline(mean_beta, color='red', linestyle='--', 
           label=f'Mean β = {mean_beta:.2f}', linewidth=2)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Transmission Rate β', fontsize=11)
ax.set_title('Transmission Rate Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# save predictions ----
df['I_pred'] = I_pred
df['S_pred'] = S_pred
df['R_pred'] = R_pred