# Discrete-time SEI-SEIR model for mosquito-borne disease (weekly)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

df = fever_df.groupby(['year', 'week']).agg({
    'total_cases': 'sum',
    'current_temperature': 'mean',
    'current_precipitation': 'mean'
}).reset_index()

df = df.sort_values(['year', 'week']).reset_index(drop=True)
print(f"Total weeks: {len(df)}")
print(f"Cases: {df['total_cases'].min():.0f} to {df['total_cases'].max():.0f}")

# parameters ----
N_h = 321992 + 500000  # human population
dt = 1  # weekly time step

# fixed biological parameters
gamma = 1/7     # human recovery rate per week (7 day infection)
sigma_h = 1/5.5 # human incubation rate per week (5.5 day latent)
mu_v = 1/2      # mosquito death rate per week (2 week lifespan)

# climate-dependent parameters
def get_params(temp, precip):
    """Calculate climate-dependent parameters"""
    # EIP decreases with temperature (7-14 days)
    EIP_days = np.clip(21 - 0.5 * temp, 7, 14)
    sigma_v = 7 / EIP_days  # per week
    
    # biting rate increases with temp (up to 30°C)
    a = np.clip(0.05 + 0.01 * temp, 0.05, 0.5)
    
    # mosquito population scales with precipitation
    precip_norm = (precip - df['current_precipitation'].min()) / \
                  (df['current_precipitation'].max() - df['current_precipitation'].min() + 0.001)
    m = 0.5 + 1.5 * precip_norm  # 0.5x to 2x
    
    return sigma_v, a, m

# fit transmission parameters from early outbreak ----
# use first 20 weeks to calibrate
calibration_weeks = min(20, len(df) // 4)
early_growth = df.iloc[:calibration_weeks]['total_cases'].values

# estimate growth rate
log_cases = np.log(early_growth + 1)
t = np.arange(len(log_cases))
growth_rate = np.polyfit(t, log_cases, 1)[0]

# use growth rate to estimate transmission strength
# for dengue, typical R0 is 2-5
target_R0 = 3.0
b = 0.5  # human infection probability
c = 0.5  # mosquito infection probability

print(f"\nUsing b={b}, c={c}, target R0={target_R0}")

# discrete-time simulation ----
n_weeks = len(df)

# initialize arrays
S_h = np.zeros(n_weeks)
E_h = np.zeros(n_weeks)
I_h = np.zeros(n_weeks)
R_h = np.zeros(n_weeks)
E_v = np.zeros(n_weeks)
I_v = np.zeros(n_weeks)

# initial conditions
I_h[0] = df.loc[0, 'total_cases']
E_h[0] = I_h[0] * 2  # assume 2x exposed
R_h[0] = 0
S_h[0] = N_h - I_h[0] - E_h[0] - R_h[0]

# initial mosquito conditions (small seed)
I_v[0] = 50
E_v[0] = 100

print("\nRunning simulation...")

# simulate forward in time
for t in range(n_weeks - 1):
    # get climate parameters
    temp = df.loc[t, 'current_temperature']
    precip = df.loc[t, 'current_precipitation']
    sigma_v, a, m = get_params(temp, precip)
    
    # mosquito population
    N_v = m * N_h * 0.5  # mosquitoes per human (0.5 to 1)
    S_v = max(0, N_v - E_v[t] - I_v[t])
    
    # force of infection (per week)
    lambda_h = a * b * I_v[t] / N_h  # to humans
    lambda_v = a * c * I_h[t] / N_h  # to mosquitoes
    
    # human transitions (discrete time)
    new_exposed_h = lambda_h * S_h[t] * dt
    new_infected_h = sigma_h * E_h[t] * dt
    new_recovered = gamma * I_h[t] * dt
    
    # mosquito transitions
    new_exposed_v = lambda_v * S_v * dt
    new_infected_v = sigma_v * E_v[t] * dt
    death_E_v = mu_v * E_v[t] * dt
    death_I_v = mu_v * I_v[t] * dt
    
    # update human compartments
    S_h[t+1] = S_h[t] - new_exposed_h
    E_h[t+1] = E_h[t] + new_exposed_h - new_infected_h
    I_h[t+1] = I_h[t] + new_infected_h - new_recovered
    R_h[t+1] = R_h[t] + new_recovered
    
    # update mosquito compartments
    E_v[t+1] = E_v[t] + new_exposed_v - new_infected_v - death_E_v
    I_v[t+1] = I_v[t] + new_infected_v - death_I_v
    
    # ensure non-negative
    S_h[t+1] = max(0, S_h[t+1])
    E_h[t+1] = max(0, E_h[t+1])
    I_h[t+1] = max(0, I_h[t+1])
    E_v[t+1] = max(0, E_v[t+1])
    I_v[t+1] = max(0, I_v[t+1])

# evaluate ----
I_pred = I_h
r2 = r2_score(df['total_cases'], I_pred)
rmse = np.sqrt(mean_squared_error(df['total_cases'], I_pred))
mae = np.mean(np.abs(df['total_cases'] - I_pred))

print(f"\n=== Model Performance ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")