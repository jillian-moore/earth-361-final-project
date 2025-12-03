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
    'current_precipitation','satellite_precipiatation',
    'current_specific_humidity',
    'vegetation_ne','vegetation_nw','vegetation_se','vegetation_sw',
    'forecast_precipitation','forecast_mean_temperature','forecast_relative_humidity'
]

corr_matrix = df[corr_vars].corr()
print(corr_matrix)

corr_matrix.to_latex(
    "figures/corr_matrix.tex",
    index=True,       # keep the is_outbreak index
    float_format="%.2f"  # round floats to 2 decimals
)

# correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
ax.set_title("Correlation Matrix for Climate, Vegetation & Cases", fontsize=14)
plt.tight_layout()
plt.savefig("output/correlation_matrix.png", dpi=300)
plt.show()

# scatter matrix for highly correlated variables
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
g.figure.subplots_adjust(top=0.92)   # <<< IMPORTANT
g.figure.suptitle(
    "Pairplot of Top Variables Correlated with Total Cases",
    fontsize=16
)
g.figure.savefig("figures/pairplot_top_correlations.png", dpi=300, bbox_inches='tight')
plt.show()

# temporal patterns ----
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

# boxplot by season
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='season', y='total_cases')
plt.title("Cases by Season")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/cases_by_season.png", dpi=300)
plt.show()

# 4. climate & vegtation relationships ----
# vegatation
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

sns.regplot(ax=axes[0], data=df, x='vegetation_se', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[0].set_title("Vegetation SE vs Cases")
axes[0].set_xlabel("Vegetation Index (NDVI)")
axes[0].set_ylabel("Total Dengue Cases")

sns.regplot(ax=axes[1], data=df, x='vegetation_sw', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[1].set_title("Vegetation SW vs Cases")
axes[1].set_xlabel("Vegetation Index (NDVI)")
axes[1].set_ylabel("Total Dengue Cases")

sns.regplot(ax=axes[2], data=df, x='vegetation_ne', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[2].set_title("Vegetation NE vs Cases")
axes[2].set_xlabel("Vegetation Index (NDVI)")
axes[2].set_ylabel("Total Dengue Cases")

sns.regplot(ax=axes[3], data=df, x='vegetation_nw', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[3].set_title("Vegetation NW vs Cases")
axes[3].set_xlabel("Vegetation Index (NDVI)")
axes[3].set_ylabel("Total Dengue Cases")

plt.tight_layout()
plt.savefig("figures/vegetation_vs_cases.png", dpi=300)
plt.show()

# current temp, precip, and humidity
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.regplot(ax=axes[0], data=df, x='current_precipitation', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[0].set_title("Current Precipitation vs Cases")
axes[0].set_xlabel("Precipitation (mm)")
axes[0].set_ylabel("Total Dengue Cases")

sns.regplot(ax=axes[1], data=df, x='current_temperature', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[1].set_title("Current Temperature vs Cases")
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("Total Dengue Cases")

sns.regplot(ax=axes[2], data=df, x='current_specific_humidity', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[2].set_title("Specific Humidity vs Cases")
axes[2].set_xlabel("Specific Humidity (g/kg)")
axes[2].set_ylabel("Total Dengue Cases")

plt.tight_layout()
plt.savefig("figures/precip_temp_humidity_vs_cases.png", dpi=300)
plt.show()

# diurnal, max, and min temp
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.regplot(ax=axes[0], data=df, x='current_diurnal_temperature_range', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[0].set_title("Diurnal Temperature Range vs Cases")
axes[0].set_xlabel("Diurnal Temp Range (°C)")
axes[0].set_ylabel("Total Dengue Cases")

sns.regplot(ax=axes[1], data=df, x='current_min_temperature', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[1].set_title("Min Temperature vs Cases")
axes[1].set_xlabel("Minimum Temperature (°C)")
axes[1].set_ylabel("Total Dengue Cases")

sns.regplot(ax=axes[2], data=df, x='current_max_temperature', y='total_cases',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[2].set_title("Max Temperature vs Cases")
axes[2].set_xlabel("Maximum Temperature (°C)")
axes[2].set_ylabel("Total Dengue Cases")

plt.tight_layout()
plt.savefig("figures/thermal_vars_vs_cases.png", dpi=300)
plt.show()

# lag analysis ----
max_lag = 6
lags = range(0, max_lag + 1)
correlations = []

for lag in lags:
    df_sorted = df.sort_values(["city","date"]).copy()
    df_sorted['precip_lag'] = df_sorted.groupby('city')['current_precipitation'].shift(lag)
    valid = df_sorted.dropna(subset=['total_cases','precip_lag'])
    correlations.append(valid['total_cases'].corr(valid['precip_lag']))

plt.figure(figsize=(8,5))
plt.plot(lags, correlations, '-o')
plt.axhline(0, color='red', linestyle='--')
plt.title("Lagged Cross-Correlation: Cases vs Precipitation")
plt.xlabel("Lag (months)")
plt.ylabel("Correlation")
plt.grid(alpha=0.3)
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
