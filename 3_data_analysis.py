# EDA

# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# load data ----
df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

# start with correlation matrix ----
corr_vars = [
    'total_cases','log_cases',
    'current_temperature','current_max_temperature','current_min_temperature',
    'current_diurnal_temperature_range',
    'current_precipitation',
    'current_specific_humidity',
    'vegetation_ne','vegetation_nw','vegetation_se','vegetation_sw',
    'forecast_precipitation','forecast_mean_temperature','forecast_relative_humidity'
]

# create abbreviated names
abbrev_names = {
    'total_cases': 'Cases',
    'log_cases': 'Log Cases',
    'current_temperature': 'Temp',
    'current_max_temperature': 'Max Temp',
    'current_min_temperature': 'Min Temp',
    'current_diurnal_temperature_range': 'DTR',
    'current_precipitation': 'Precip',
    'current_specific_humidity': 'Humidity',
    'vegetation_ne': 'Veg NE',
    'vegetation_nw': 'Veg NW',
    'vegetation_se': 'Veg SE',
    'vegetation_sw': 'Veg SW',
    'forecast_precipitation': 'Fcst Precip',
    'forecast_mean_temperature': 'Fcst Temp',
    'forecast_relative_humidity': 'Fcst RH'
}

corr_matrix = df[corr_vars].corr()
corr_matrix = corr_matrix.rename(columns=abbrev_names, index=abbrev_names)

corr_matrix.to_latex(
    "figures/corr_matrix.tex",
    index=False,
    escape=False,
    float_format="%.2f"
)

# correlation heatmap
corr_matrix = df[corr_vars].corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
ax.set_title("Correlation Matrix for Climate, Vegetation & Cases", fontsize=14)
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=300)
plt.show()

# scatter matrix for highly correlated variables
corr_matrix = df[corr_vars].corr()
top_corr = (
    corr_matrix['total_cases']
    .abs()
    .sort_values(ascending=False)
    .iloc[1:6]   # skip total_cases itself, take next top variables
    .index
    .tolist()
)

# generate pairplot
g = sns.pairplot(df[top_corr], diag_kind='kde')
g.figure.subplots_adjust(top=0.92)  
g.figure.suptitle(
    "Pairplot of Top Variables Correlated with Total Cases",
    fontsize=16
)
g.figure.savefig("figures/pairplot_top_correlations.png", dpi=300, bbox_inches='tight')
plt.show()

# temporal patterns ----
# everyone
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

monthly = df.groupby(['year', 'month']).agg({'total_cases':'sum'}).reset_index()
monthly['date'] = pd.to_datetime(monthly[['year','month']].assign(day=1))

plt.figure(figsize=(12,5))
plt.plot(monthly['date'], monthly['total_cases'], '-o')
plt.title("Monthly Total Cases Over Time", fontsize=13)
plt.ylabel("Total Dengue Cases")
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/monthly_cases.png", dpi=300)
plt.show()

# san juan
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

df_city0 = df[df['city'] == 0].copy()

monthly0 = (
    df_city0.groupby(['year', 'month'])['total_cases']
    .sum()
    .reset_index()
)

monthly0['date'] = pd.to_datetime(
    monthly0[['year', 'month']].assign(day=1)
)

title_size = 20
label_size = 18
tick_size  = 14

plt.figure(figsize=(12, 6))
plt.plot(
    monthly0['date'],
    monthly0['total_cases'],
    '-o',
    markersize=6,
    linewidth=2.5
)

plt.title("Monthly Dengue Cases Over Time (San Juan, Puerto Rico)", fontsize=title_size)
plt.ylabel("Total Dengue Cases", fontsize=label_size)
plt.xlabel("Date", fontsize=label_size)

plt.grid(alpha=0.3)
plt.xticks(rotation=45, fontsize=tick_size)
plt.yticks(fontsize=tick_size)

plt.tight_layout()
plt.savefig("figures/monthly_cases_city0.png", dpi=400, bbox_inches='tight')
plt.show()

# boxplot by season
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='season', y='total_cases')
plt.title("Cases by Season")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/cases_by_season.png", dpi=300)
plt.show()

# 4. climate & vegetation relationships ----
title_size = 18
label_size = 16
tick_size = 14

fig, axes = plt.subplots(2, 2, figsize=(18, 10))  # Taller height for 2x2 grid

# SE
sns.regplot(
    ax=axes[0, 0],
    data=df,
    x='vegetation_se',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)
axes[0, 0].set_title("Vegetation SE vs Cases", fontsize=title_size)
axes[0, 0].set_xlabel("Vegetation Index (NDVI)", fontsize=label_size)
axes[0, 0].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[0, 0].tick_params(axis='both', labelsize=tick_size)

# SW
sns.regplot(
    ax=axes[0, 1],
    data=df,
    x='vegetation_sw',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)
axes[0, 1].set_title("Vegetation SW vs Cases", fontsize=title_size)
axes[0, 1].set_xlabel("Vegetation Index (NDVI)", fontsize=label_size)
axes[0, 1].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[0, 1].tick_params(axis='both', labelsize=tick_size)

# NE
sns.regplot(
    ax=axes[1, 0],
    data=df,
    x='vegetation_ne',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)
axes[1, 0].set_title("Vegetation NE vs Cases", fontsize=title_size)
axes[1, 0].set_xlabel("Vegetation Index (NDVI)", fontsize=label_size)
axes[1, 0].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[1, 0].tick_params(axis='both', labelsize=tick_size)

# NW
sns.regplot(
    ax=axes[1, 1],
    data=df,
    x='vegetation_nw',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)
axes[1, 1].set_title("Vegetation NW vs Cases", fontsize=title_size)
axes[1, 1].set_xlabel("Vegetation Index (NDVI)", fontsize=label_size)
axes[1, 1].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[1, 1].tick_params(axis='both', labelsize=tick_size)

plt.tight_layout()
plt.savefig("figures/vegetation_vs_cases.png", dpi=300)
plt.show()

# current temp, precip, and humidity
title_size = 18
label_size = 16
tick_size = 14

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.regplot(
    ax=axes[0],
    data=df,
    x='current_precipitation',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)

axes[0].set_title("Current Precipitation vs Cases", fontsize=title_size)
axes[0].set_xlabel("Precipitation (mm)", fontsize=label_size)
axes[0].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[0].tick_params(axis='both', labelsize=tick_size)

sns.regplot(
    ax=axes[1],
    data=df,
    x='current_temperature',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)

axes[1].set_title("Current Temperature vs Cases", fontsize=title_size)
axes[1].set_xlabel("Temperature (°C)", fontsize=label_size)
axes[1].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[1].tick_params(axis='both', labelsize=tick_size)

sns.regplot(
    ax=axes[2],
    data=df,
    x='current_specific_humidity',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)

axes[2].set_title("Specific Humidity vs Cases", fontsize=title_size)
axes[2].set_xlabel("Specific Humidity (g/kg)", fontsize=label_size)
axes[2].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[2].tick_params(axis='both', labelsize=tick_size)

plt.tight_layout()
plt.savefig("figures/precip_temp_humidity_vs_cases.png", dpi=300)
plt.show()

# diurnal, max, and min temp
# set font sizes
title_size = 18
label_size = 16
tick_size = 14

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.regplot(
    ax=axes[0],
    data=df,
    x='current_diurnal_temperature_range',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)
axes[0].set_title("Diurnal Temperature Range vs Cases", fontsize=title_size)
axes[0].set_xlabel("Diurnal Temp Range (°C)", fontsize=label_size)
axes[0].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[0].tick_params(axis='both', labelsize=tick_size)

sns.regplot(
    ax=axes[1],
    data=df,
    x='current_min_temperature',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)
axes[1].set_title("Min Temperature vs Cases", fontsize=title_size)
axes[1].set_xlabel("Minimum Temperature (°C)", fontsize=label_size)
axes[1].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[1].tick_params(axis='both', labelsize=tick_size)

sns.regplot(
    ax=axes[2],
    data=df,
    x='current_max_temperature',
    y='total_cases',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)
axes[2].set_title("Max Temperature vs Cases", fontsize=title_size)
axes[2].set_xlabel("Maximum Temperature (°C)", fontsize=label_size)
axes[2].set_ylabel("Total Dengue Cases", fontsize=label_size)
axes[2].tick_params(axis='both', labelsize=tick_size)

plt.tight_layout()
plt.savefig("figures/thermal_vars_vs_cases.png", dpi=300)
plt.show()

# lag analysis ----
max_lag = 6
lags = range(0, max_lag + 1)
correlations = []

# compute lagged correlations
for lag in lags:
    df_sorted = df.sort_values(["city", "date"]).copy()
    df_sorted['precip_lag'] = df_sorted.groupby('city')['satellite_precipiatation'].shift(lag)
    valid = df_sorted.dropna(subset=['total_cases', 'precip_lag'])
    correlations.append(valid['total_cases'].corr(valid['precip_lag']))

title_size = 18
label_size = 16
tick_size = 14

plt.figure(figsize=(10, 6))
plt.plot(lags, correlations, '-o', markersize=8, linewidth=2)

plt.axhline(0, color='red', linestyle='--', linewidth=1.5)

plt.title("Lagged Cross-Correlation: Cases vs Precipitation", fontsize=title_size)
plt.xlabel("Lag (months)", fontsize=label_size)
plt.ylabel("Correlation", fontsize=label_size)

plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/lag_correlation_precip.png", dpi=300)
plt.show()

# outbreak analysis ----
outbreak_counts = df.groupby('city')['is_outbreak'].sum().sort_values(ascending=False)

plt.figure(figsize=(10,6))
outbreak_counts.head(10).plot(kind='barh', color='crimson')
plt.title("Outbreak Months by City")
plt.xlabel("Number of Outbreak Months")
plt.ylabel("City (0: San Juan, 1: Iquitos)")
plt.tight_layout()
plt.savefig("figures/outbreaks_by_city.png", dpi=300)
plt.show()

# climate conditions during outbreaks
outbreak_summary = df.groupby('is_outbreak')[[
    'current_temperature','current_precipitation',
    'current_specific_humidity','vegetation_ne'
]].mean()

outbreak_summary.to_latex(
    "figures/outbreak_summary.tex",
    index=True,       # keep the is_outbreak index
    float_format="%.2f"  # round floats to 2 decimals
)
