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

# After the simulation loop, add:

# Calculate mosquito populations over time
m_values = []
N_v_values = []
for t in range(n_weeks):
    temp = df.loc[t, 'current_temperature']
    precip = df.loc[t, 'current_precipitation']
    _, _, m = get_params(temp, precip)
    m_values.append(m)
    N_v_values.append(m * N_h * 0.5)

# Add a new plot
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Temperature and precipitation
ax = axes2[0]
ax2 = ax.twinx()
ax.plot(df['current_temperature'], color='red', label='Temperature')
ax2.plot(df['current_precipitation'], color='blue', label='Precipitation', alpha=0.6)
ax.set_ylabel('Temperature (°C)', color='red')
ax2.set_ylabel('Precipitation (mm)', color='blue')
ax.set_title('Climate Variables')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot 2: Mosquito multiplier
ax = axes2[1]
ax.plot(m_values, color='green', linewidth=2)
ax.set_ylabel('Mosquito Multiplier (m)')
ax.set_title('Climate-Based Mosquito Population Multiplier')
ax.axhline(1.0, color='black', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)

# Plot 3: Total mosquito population vs infection prevalence
ax = axes2[2]
ax.plot(N_v_values, label='Total Mosquito Pop', color='gray', linewidth=2)
ax.plot(I_v, label='Infected Mosquitoes', color='red', linewidth=2)
ax.set_ylabel('Count')
ax.set_xlabel('Week')
ax.set_title('Mosquito Population vs Infected')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')  # log scale to see both

plt.tight_layout()
plt.savefig('mosquito_diagnostics.png', dpi=300)
plt.show()

# Print diagnostics
print(f"\n=== Mosquito Diagnostics ===")
print(f"Mosquito multiplier range: {min(m_values):.2f} to {max(m_values):.2f}")
print(f"Total mosquito population range: {min(N_v_values):.0f} to {max(N_v_values):.0f}")
print(f"Initial infected prevalence: {I_v[0]/N_v_values[0]*100:.4f}%")
print(f"Max infected mosquitoes: {max(I_v):.0f}")
print(f"Weeks with <10 infected mosquitoes: {sum(I_v < 10)}")

# evaluate ----
I_pred = I_h
r2 = r2_score(df['total_cases'], I_pred)
rmse = np.sqrt(mean_squared_error(df['total_cases'], I_pred))
mae = np.mean(np.abs(df['total_cases'] - I_pred))

print(f"\n=== Model Performance ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# visualize ----
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# plot 1: cases
ax = axes[0, 0]
ax.plot(df['total_cases'].values, 'o-', label='Observed', 
        linewidth=2, markersize=4, color='red', alpha=0.7)
ax.plot(I_pred, '-', label='Predicted (SEI-SEIR)', 
        linewidth=2, color='blue', alpha=0.8)
ax.set_xlabel('Week', fontsize=11)
ax.set_ylabel('Cases', fontsize=11)
ax.set_title(f'Dengue Cases (R²={r2:.3f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# plot 2: human compartments
ax = axes[0, 1]
ax.plot(S_h/N_h, label='Susceptible', linewidth=2, alpha=0.7)
ax.plot(E_h/N_h, label='Exposed', linewidth=2, alpha=0.7)
ax.plot(I_h/N_h, label='Infected', linewidth=2, alpha=0.7)
ax.plot(R_h/N_h, label='Recovered', linewidth=2, alpha=0.7)
ax.set_xlabel('Week', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Human Compartments', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# plot 3: mosquito dynamics
ax = axes[1, 0]
ax.plot(E_v, label='Exposed Mosquitoes', linewidth=2, alpha=0.7, color='orange')
ax.plot(I_v, label='Infected Mosquitoes', linewidth=2, alpha=0.7, color='red')
ax.set_xlabel('Week', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Mosquito Infection Dynamics', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# plot 4: residuals
ax = axes[1, 1]
residuals = df['total_cases'].values - I_pred
ax.plot(residuals, 'o-', linewidth=2, markersize=4, color='purple', alpha=0.6)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.fill_between(range(len(residuals)), residuals, 0, alpha=0.2, color='purple')
ax.set_xlabel('Week', fontsize=11)
ax.set_ylabel('Residual', fontsize=11)
ax.set_title(f'Residuals (RMSE={rmse:.1f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sei_seir_discrete.png', dpi=300, bbox_inches='tight')
plt.show()