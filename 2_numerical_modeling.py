# numerical modeling: SIR model with climate-dependent transmission (WEEKLY)

# load packages ----
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# aggregate to weekly totals ----
df = fever_df.groupby(['year', 'week']).agg({
    'total_cases': 'sum',
    'current_temperature': 'mean',
    'current_precipitation': 'mean'
}).reset_index()

df = df.sort_values(['year', 'week']).reset_index(drop=True)
print(df[['current_temperature','current_precipitation']].describe())

# define human + mosquito parameters ----
N_h = 321992 + 500000      # human population
gamma = 1/7                # human recovery rate per week
sigma_h = 1/1              # human incubation rate per week (latent period ~1 week)
sigma_v_base = 1/1         # mosquito incubation rate (1/EIP) per week
mu_v_base = 0.1            # mosquito death rate per week
a_base = 0.3               # biting rate per mosquito per week
b = 0.5                     # human infection prob
c = 0.5                     # mosquito infection prob
N_v = 1e5                   # assume fixed mosquito population (can scale with rainfall)

# fit regression: mosquito force of infection proxy ----
# we'll model beta ~ climate as a driver for mosquito infection
df['total_cases_lag'] = df['total_cases'].shift(1).fillna(0)
X = df[['current_temperature', 'current_precipitation']]
y = df['total_cases_lag'] / N_h   # rough proxy for mosquito infection rate
mosq_model = LinearRegression().fit(X, y)

def mosq_infection(t_idx):
    """proxy mosquito infection from climate"""
    T = df.loc[t_idx, 'current_temperature']
    R = df.loc[t_idx, 'current_precipitation']
    return max(0.001, mosq_model.intercept_ + mosq_model.coef_[0]*T + mosq_model.coef_[1]*R)

# define SEI-SEIR ODE system ----
def sei_seir_ode(t, y):
    S_h, E_h, I_h, R_h, S_v, E_v, I_v = y
    t_idx = int(np.clip(round(t), 0, len(df)-1))
    
    # mosquito climate-driven rates
    a = a_base  # biting rate, could scale with T
    mu_v = mu_v_base  # mosquito death rate
    sigma_v = sigma_v_base  # mosquito incubation
    
    # force of infection
    lambda_h = a * b * I_v / N_h          # humans
    lambda_v = a * c * I_h / N_h + mosq_infection(t_idx)  # mosquitoes, plus climate-driven
    
    # human equations
    dS_h = -lambda_h * S_h
    dE_h = lambda_h * S_h - sigma_h * E_h
    dI_h = sigma_h * E_h - gamma * I_h
    dR_h = gamma * I_h
    
    # mosquito equations
    dS_v = -lambda_v * S_v - mu_v * S_v
    dE_v = lambda_v * S_v - sigma_v * E_v - mu_v * E_v
    dI_v = sigma_v * E_v - mu_v * I_v
    
    return [dS_h, dE_h, dI_h, dR_h, dS_v, dE_v, dI_v]

# initial conditions ----
I0 = df.loc[0, 'total_cases']
E0 = I0 * 0.5
R0 = 0
S0 = N_h - I0 - R0 - E0

# mosquito initial conditions
I_v0 = 100
E_v0 = 1000
S_v0 = N_v - I_v0 - E_v0

y0 = [S0, E0, I0, R0, S_v0, E_v0, I_v0]

# solve ODE ----
t_span = [0, len(df)-1]
t_eval = np.arange(len(df))

sol = solve_ivp(sei_seir_ode, t_span, y0, t_eval=t_eval, method='RK45')

# extract predictions ----
S_h_pred, E_h_pred, I_h_pred, R_h_pred = sol.y[:4]

# evaluate model ----
r2 = r2_score(df['total_cases'], I_h_pred)
rmse = np.sqrt(mean_squared_error(df['total_cases'], I_h_pred))
mae = np.mean(np.abs(df['total_cases'] - I_h_pred))

print(f"\n=== SEI-SEIR Model Performance ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# plot results ----
plt.figure(figsize=(12,5))
plt.plot(df['total_cases'], label='Observed Cases', color='red')
plt.plot(I_h_pred, label='Predicted Cases (SEI-SEIR)', color='blue')
plt.xlabel('Week')
plt.ylabel('Cases')
plt.legend()
plt.title('SEI-SEIR Dengue Model')
plt.show()
