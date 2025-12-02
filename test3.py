# SEI-SEIR simulation

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# --- load data
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

df = fever_df.groupby(['year', 'week']).agg({
    'total_cases': 'sum',
    'current_temperature': 'mean',
    'current_precipitation': 'mean'
}).reset_index()

df = df.sort_values(['year', 'week']).reset_index(drop=True)
n_weeks = len(df)
print(f"Total weeks: {n_weeks}, Cases range: {df['total_cases'].min()} - {df['total_cases'].max()}")

# --- parameters
N_h = 321992 + 500000  # human population

# biological rates (per week)
gamma = 1.4       # human recovery
sigma_h = 1.4    # human incubation
mu_v = 1/2         # mosquito death

b = 0.7  # human infection probability
c = 0.7  # mosquito infection probability

# small time step for stability
dt = 0.2  # week (~1.4 days)
n_steps = int(n_weeks / dt)
time = np.arange(0, n_weeks, dt)

# --- climate-dependent parameters
def get_params(temp, precip):
    EIP_days = np.clip(21 - 0.5 * temp, 7, 14)
    sigma_v = 7 / EIP_days  # per week
    a = np.clip(0.05 + 0.01 * temp, 0.05, 0.5)
    
    # normalized precipitation for mosquito multiplier
    precip_norm = (precip - df['current_precipitation'].min()) / \
                  (df['current_precipitation'].max() - df['current_precipitation'].min() + 1e-6)
    m = 0.5 + 1.5 * precip_norm
    return sigma_v, a, m

# interpolate climate for fractional time steps
from scipy.interpolate import interp1d
temp_func = interp1d(np.arange(n_weeks), df['current_temperature'].values, kind='linear', fill_value='extrapolate')
precip_func = interp1d(np.arange(n_weeks), df['current_precipitation'].values, kind='linear', fill_value='extrapolate')

# initialize compartments ---
S_h = np.zeros(n_steps)
E_h = np.zeros(n_steps)
I_h = np.zeros(n_steps)
R_h = np.zeros(n_steps)
S_v = np.zeros(n_steps)
E_v = np.zeros(n_steps)
I_v = np.zeros(n_steps)

# initial human conditions
I_h[0] = df.loc[0, 'total_cases']
E_h[0] = 0.5 * I_h[0]
R_h[0] = 0
S_h[0] = N_h - I_h[0] - E_h[0] - R_h[0]

# initial mosquito conditions using first week climate
temp0 = temp_func(0)
precip0 = precip_func(0)
sigma_v0, a0, m0 = get_params(temp0, precip0)
N_v0 = m0 * N_h

# start with finding mosquito infections
required_I_v = max(1, (I_h[0] * N_h) / (a0 * b * S_h[0]))
I_v[0] = required_I_v
E_v[0] = I_v[0] * 1.5
S_v[0] = max(0, N_v0 - I_v[0] - E_v[0])

# simulation loop ---
for t in range(n_steps - 1):
    temp = temp_func(t*dt)
    precip = precip_func(t*dt)
    sigma_v, a, m = get_params(temp, precip)
    N_v_t = m * N_h

    # effective susceptible mosquitoes
    S_v_eff = max(0, S_v[t] + 0.1 * mu_v * N_v_t * dt)

    # force of infection
    lambda_h = a * b * I_v[t] / N_h
    lambda_v = a * c * I_h[t] / N_h

    # human updates
    new_E_h = lambda_h * S_h[t] * dt
    new_I_h = sigma_h * E_h[t] * dt
    new_R_h = gamma * I_h[t] * dt

    S_h[t+1] = max(0, S_h[t] - new_E_h)
    E_h[t+1] = max(0, E_h[t] + new_E_h - new_I_h)
    I_h[t+1] = max(0, I_h[t] + new_I_h - new_R_h)
    R_h[t+1] = max(0, R_h[t] + new_R_h)

    # mosquito updates
    new_E_v = lambda_v * S_v_eff * dt
    new_I_v = sigma_v * E_v[t] * dt
    death_S = mu_v * S_v_eff * dt
    death_E = mu_v * E_v[t] * dt
    death_I = mu_v * I_v[t] * dt

    S_v[t+1] = max(0, S_v_eff - new_E_v - death_S)
    E_v[t+1] = max(0, E_v[t] + new_E_v - new_I_v - death_E)
    I_v[t+1] = max(0, I_v[t] + new_I_v - death_I)

# --- sample weekly output
weekly_indices = (np.arange(n_weeks) / dt).astype(int)
I_pred = I_h[weekly_indices]

# --- evaluate
r2 = r2_score(df['total_cases'], I_pred)
rmse = np.sqrt(mean_squared_error(df['total_cases'], I_pred))
mae = np.mean(np.abs(df['total_cases'] - I_pred))

print(f"R²: {r2:.3f}, RMSE: {rmse:.1f}, MAE: {mae:.1f}")

# --- plot
plt.figure(figsize=(12,5))
plt.plot(df['total_cases'], 'o-', label='Observed', color='red')
plt.plot(I_pred, '-', label='Predicted', color='blue')
plt.xlabel('Week')
plt.ylabel('Cases')
plt.legend()
plt.grid(True)
plt.show()
