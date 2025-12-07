# SEI-SEIR model for dengue (vector-borne disease)

# load packages ----
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# load data ----
df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# filter by city and years 1994-1996
city_value = 0
df_city = df[(df['city'] == city_value) & 
             (df['year'] >= 1994) & 
             (df['year'] <= 1996)].copy()

# aggregate by week
df_city = df_city.groupby(['year', 'week']).agg({'total_cases': 'sum'}).reset_index()
df_city = df_city.sort_values(['year', 'week']).reset_index(drop=True)

observed_cases = df_city['total_cases'].values
n_weeks = len(observed_cases)

# population parameters ----
N_h = 3201000  # total human population

# fixed model parameters ----
# human parameters
sigma_h = 1.4   # human incubation rate (1/latent period) per week
gamma_h = 1.0   # human recovery rate (1/infectious period) per week

# mosquito parameters
mu_v = 0.5      # mosquito death rate per week
sigma_v = 1.0   # mosquito incubation rate (1/EIP) per week
sigma_bite = 1.0  # mosquito biting rate per week

# transmission probabilities
p_h = 0.5       # probability human infection per infectious bite
p_v = 0.5       # probability mosquito infection per bite on infectious human

# derived transmission rates
beta_h = p_h * sigma_bite  # human infection rate
beta_v = p_v * sigma_bite  # mosquito infection rate

# mosquito-to-human ratio
m = 20         # mosquitoes per human
N_v = m * N_h   # total mosquito population

# print parameters ----
print(f"Human population N_h: {N_h:,}")
print(f"Mosquito population N_v: {N_v:,}")
print(f"Mosquito-to-human ratio m: {m}")
print(f"Human recovery rate gamma_h: {gamma_h} per week")
print(f"Human incubation rate sigma_h: {sigma_h} per week")
print(f"Mosquito death rate mu_v: {mu_v} per week")
print(f"Mosquito incubation rate sigma_v: {sigma_v} per week")
print(f"Transmission probability (human) p_h: {p_h}")
print(f"Transmission probability (mosquito) p_v: {p_v}")

# SEI-SEIR model equations ----
def sei_seir_model(t, y):
    """
    SEI-SEIR model for vector-borne disease (dengue)
    
    state variables (as proportions):
        mosquito (vector): s_v, e_v, i_v
        human (host): s_h, e_h, i_h, r_h
    
    human equations:
        ds_h/dt = -beta_h * i_v * s_h
        de_h/dt = beta_h * i_v * s_h - sigma_h * e_h
        di_h/dt = sigma_h * e_h - gamma_h * i_h
        dr_h/dt = gamma_h * i_h
    
    mosquito equations:
        ds_v/dt = B_v - beta_v * i_h * s_v - mu_v * s_v
        de_v/dt = beta_v * i_h * s_v - sigma_v * e_v - mu_v * e_v
        di_v/dt = sigma_v * e_v - mu_v * i_v
    
    where B_v = mu_v (birth rate = death rate for equilibrium)
    """
    s_v, e_v, i_v, s_h, e_h, i_h, r_h = y
    
    # mosquito birth rate (maintains population)
    B_v = mu_v
    
    # human equations ----
    ds_h = -beta_h * i_v * s_h
    de_h = beta_h * i_v * s_h - sigma_h * e_h
    di_h = sigma_h * e_h - gamma_h * i_h
    dr_h = gamma_h * i_h
    
    # mosquito equations ----
    ds_v = B_v - beta_v * i_h * s_v - mu_v * s_v
    de_v = beta_v * i_h * s_v - sigma_v * e_v - mu_v * e_v
    di_v = sigma_v * e_v - mu_v * i_v
    
    return [ds_v, de_v, di_v, ds_h, de_h, di_h, dr_h]

# initial conditions ----
# human initial conditions (as proportions)
i_h0 = observed_cases[0] / N_h  # initial infected proportion
e_h0 = 2 * i_h0                  # initial exposed (assume 2x infected)
r_h0 = 0.0                       # no recovered initially
s_h0 = 1.0 - e_h0 - i_h0 - r_h0  # remaining are susceptible

# mosquito initial conditions (as proportions)
# estimate infected mosquitoes needed to sustain human infections
i_v0 = (sigma_h + gamma_h) * i_h0 / (beta_h * s_h0)
i_v0 = min(i_v0, 0.1)  # cap at 10%
e_v0 = 2 * i_v0        # exposed mosquitoes
s_v0 = 1.0 - e_v0 - i_v0

y0 = [s_v0, e_v0, i_v0, s_h0, e_h0, i_h0, r_h0]

# compute R0 ----
# for vector-borne diseases, R0 = sqrt(product of two transmission cycles)
R0_human = beta_h * sigma_v / (mu_v * (sigma_v + mu_v))
R0_vector = beta_v * sigma_h / (gamma_h * mu_v)
R0 = np.sqrt(R0_human * R0_vector)

if R0 > 1:
    print("  R0 > 1: epidemic conditions present")
else:
    print("  R0 < 1: disease will die out")

# solve ODE system using solve_ivp ----
t_span = (0, n_weeks)
t_eval = np.arange(n_weeks)

sol = solve_ivp(
    sei_seir_model,
    t_span,
    y0,
    t_eval=t_eval,
    method='LSODA',  # good for stiff equations
    rtol=1e-7,
    atol=1e-9
)

if sol.success:
    print("✓ Integration successful!")
else:
    print(f"✗ Integration failed: {sol.message}")

# extract solution ----
s_v, e_v, i_v, s_h, e_h, i_h, r_h = sol.y

# convert infected proportion to actual cases
predicted_cases = i_h * N_h

# evaluate model ----
r2 = r2_score(observed_cases, predicted_cases)
rmse = np.sqrt(mean_squared_error(observed_cases, predicted_cases))
mae = mean_absolute_error(observed_cases, predicted_cases)

print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.1f}")
print(f"MAE = {mae:.1f}")

# save results ----
results_df = pd.DataFrame({
    'week': np.arange(n_weeks),
    'observed_cases': observed_cases,
    'predicted_cases': predicted_cases,
    's_h': s_h,
    'e_h': e_h,
    'i_h': i_h,
    'r_h': r_h,
    's_v': s_v,
    'e_v': e_v,
    'i_v': i_v
})
results_df.to_csv('results/seir_1994_1996_results.csv', index=False)

# save parameters ----
params_df = pd.DataFrame([{
    'city': city_value,
    'year_start': 1994,
    'year_end': 1996,
    'N_h': N_h,
    'N_v': N_v,
    'm': m,
    'sigma_h': sigma_h,
    'gamma_h': gamma_h,
    'mu_v': mu_v,
    'sigma_v': sigma_v,
    'p_h': p_h,
    'p_v': p_v,
    'beta_h': beta_h,
    'beta_v': beta_v,
    'R0': R0,
    'r2': r2,
    'rmse': rmse,
    'mae': mae
}])
params_df.to_csv('results/seir_1994_1996_parameters.csv', index=False)

# visualizations ----
title_size = 20
label_size = 18
tick_size  = 14
legend_size = 14

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# observed vs predicted cases
ax1 = axes[0]

ax1.plot(
    observed_cases, 'o',
    label='Observed',
    color='red',
    alpha=0.6,
    markersize=4
)
ax1.plot(
    predicted_cases, '-',
    label='Predicted',
    color='blue',
    linewidth=2.5
)

ax1.set_xlabel('Week', fontsize=label_size)
ax1.set_ylabel('Cases', fontsize=label_size)
ax1.set_title(
    f'Observed vs Predicted (1994–1996)\n'
    f'$R^2$={r2:.3f},  RMSE={rmse:.1f},  $R_0$={R0:.2f}',
    fontsize=title_size
)

ax1.tick_params(axis='both', labelsize=tick_size)
ax1.legend(fontsize=legend_size)
ax1.grid(True, alpha=0.3)

# humans/mosquitoes exposed/`infected
ax2 = axes[1]

ax2.plot(e_h, label='Human Exposed ($E_h$)',
         color='orange', linestyle='--', linewidth=2.5)
ax2.plot(i_h, label='Human Infected ($I_h$)',
         color='red', linewidth=2.5)
ax2.plot(e_v, label='Mosquito Exposed ($E_v$)',
         color='gold', linestyle='--', linewidth=2, alpha=0.8)
ax2.plot(i_v, label='Mosquito Infected ($I_v$)',
         color='darkred', linewidth=2, alpha=0.8)

ax2.set_xlabel('Week', fontsize=label_size)
ax2.set_ylabel('Proportion', fontsize=label_size)
ax2.set_title('Human vs Mosquito\nExposed & Infected', fontsize=title_size)

ax2.tick_params(axis='both', labelsize=tick_size)
ax2.legend(fontsize=legend_size)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/seir_side_by_side.png', dpi=400, bbox_inches='tight')
plt.show()
