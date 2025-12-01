# exploratory data analysis

# load packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# load data ----
fever_df = pd.read_csv("data/processed/dengue_data_cleaned.csv")

print(f"Data shape: {fever_df.shape}")
print(f"Date range: {fever_df['year'].min()}-{fever_df['year'].max()}")
print(f"Number of districts: {fever_df['district'].nunique()}")

# basic statistics ----
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(fever_df[['cases', 'temp_avg', 'precipitation_avg', 'humidity_avg']].describe())

# ============================================
# 1. TEMPORAL PATTERNS
# ============================================

# aggregate to national monthly totals
monthly_national = fever_df.groupby(['year', 'month']).agg({
    'cases': 'sum',
    'temp_avg': 'mean',
    'precipitation_avg': 'mean',
    'humidity_avg': 'mean'
}).reset_index()

monthly_national = monthly_national.sort_values(['year', 'month']).reset_index(drop=True)
monthly_national['date'] = pd.to_datetime(monthly_national[['year', 'month']].assign(day=1))

# line plot: cases over time ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly_national['date'], monthly_national['cases'], 
        'o-', linewidth=2, markersize=6, color='steelblue')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Total Cases', fontsize=11)
ax.set_title('Dengue Cases Over Time (National)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/cases_over_time.png', dpi=300, bbox_inches='tight')
plt.show()

# box plot: cases by month ----
fig, ax = plt.subplots(figsize=(10, 5))
fever_df.boxplot(column='cases', by='month', ax=ax)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Cases', fontsize=11)
ax.set_title('Distribution of Cases by Month', fontsize=12, fontweight='bold')
plt.suptitle('')  # remove default title
plt.tight_layout()
plt.savefig('output/cases_by_month_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# box plot: cases by season ----
fig, ax = plt.subplots(figsize=(10, 5))
fever_df.boxplot(column='cases', by='season', ax=ax)
ax.set_xlabel('Season', fontsize=11)
ax.set_ylabel('Cases', fontsize=11)
ax.set_title('Distribution of Cases by Season', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/cases_by_season_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. REGIONAL PATTERNS
# ============================================

# aggregate by district
district_summary = fever_df.groupby('district').agg({
    'cases': ['sum', 'mean', 'std'],
    'temp_avg': 'mean',
    'precipitation_avg': 'mean',
    'latitude': 'first',
    'longitude': 'first'
}).reset_index()

district_summary.columns = ['district', 'total_cases', 'mean_cases', 'std_cases',
                             'avg_temp', 'avg_precip', 'latitude', 'longitude']

# top 10 districts by total cases
top_districts = district_summary.nlargest(10, 'total_cases')

print("\n" + "="*50)
print("TOP 10 DISTRICTS BY TOTAL CASES")
print("="*50)
print(top_districts[['district', 'total_cases', 'mean_cases']])

# bar plot: top 10 districts ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_districts['district'], top_districts['total_cases'], color='coral')
ax.set_xlabel('Total Cases (2019-2021)', fontsize=11)
ax.set_ylabel('District', fontsize=11)
ax.set_title('Top 10 Districts by Total Dengue Cases', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('output/top_districts_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

# box plot: cases by province ----
fig, ax = plt.subplots(figsize=(12, 6))
fever_df.boxplot(column='cases', by='province', ax=ax)
ax.set_xlabel('Province', fontsize=11)
ax.set_ylabel('Cases', fontsize=11)
ax.set_title('Distribution of Cases by Province', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/cases_by_province_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 3. CLIMATE RELATIONSHIPS
# ============================================

# scatter plots: cases vs climate variables ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# temperature vs cases
axes[0].scatter(fever_df['temp_avg'], fever_df['cases'], alpha=0.5, color='coral')
axes[0].set_xlabel('Average Temperature (°C)', fontsize=10)
axes[0].set_ylabel('Cases', fontsize=10)
axes[0].set_title('Temperature vs Cases', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# precipitation vs cases
axes[1].scatter(fever_df['precipitation_avg'], fever_df['cases'], alpha=0.5, color='steelblue')
axes[1].set_xlabel('Average Precipitation (mm)', fontsize=10)
axes[1].set_ylabel('Cases', fontsize=10)
axes[1].set_title('Precipitation vs Cases', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# humidity vs cases
axes[2].scatter(fever_df['humidity_avg'], fever_df['cases'], alpha=0.5, color='forestgreen')
axes[2].set_xlabel('Average Humidity (%)', fontsize=10)
axes[2].set_ylabel('Cases', fontsize=10)
axes[2].set_title('Humidity vs Cases', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/climate_vs_cases_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# correlation analysis ----
climate_vars = ['cases', 'temp_avg', 'precipitation_avg', 'humidity_avg']
corr_matrix = fever_df[climate_vars].corr()

print("\n" + "="*50)
print("CORRELATION MATRIX")
print("="*50)
print(corr_matrix)

# correlation heatmap ----
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Between Cases and Climate Variables', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 4. GEOSPATIAL MAPPING
# ============================================

# simple scatter plot map ----
# size by total cases, color by average temperature
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(district_summary['longitude'], 
                     district_summary['latitude'],
                     s=district_summary['total_cases']/10,  # scale for visibility
                     c=district_summary['avg_temp'],
                     cmap='RdYlBu_r',
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5)

ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title('Dengue Cases by District (Size = Total Cases, Color = Temperature)', 
             fontsize=12, fontweight='bold')

# add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Average Temperature (°C)', fontsize=10)

# annotate top 5 districts
for _, row in top_districts.head(5).iterrows():
    ax.annotate(row['district'], 
                (row['longitude'], row['latitude']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/geospatial_map.png', dpi=300, bbox_inches='tight')
plt.show()

# heatmap-style visualization using pivot ----
# create a grid of cases by district and time
pivot_data = fever_df.pivot_table(
    values='cases',
    index='district',
    columns='month',
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Average Cases'})
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('District', fontsize=11)
ax.set_title('Average Cases by District and Month', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('output/district_month_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 5. CROSS-CORRELATION ANALYSIS
# ============================================

# calculate lagged correlation: cases vs precipitation
max_lag = 3
lags = range(0, max_lag + 1)
correlations = []

for lag in lags:
    # shift precipitation by lag months
    fever_df_sorted = fever_df.sort_values(['district', 'year', 'month'])
    fever_df_sorted['precip_lag'] = fever_df_sorted.groupby('district')['precipitation_avg'].shift(lag)
    
    # calculate correlation
    valid_data = fever_df_sorted.dropna(subset=['cases', 'precip_lag'])
    if len(valid_data) > 0:
        corr = valid_data['cases'].corr(valid_data['precip_lag'])
        correlations.append(corr)
    else:
        correlations.append(np.nan)

# plot cross-correlation ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lags, correlations, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Lag (months)', fontsize=11)
ax.set_ylabel('Correlation Coefficient', fontsize=11)
ax.set_title('Cross-Correlation: Cases vs Lagged Precipitation', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/cross_correlation_lag.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("LAGGED CORRELATIONS (Cases vs Precipitation)")
print("="*50)
for lag, corr in zip(lags, correlations):
    print(f"Lag {lag} months: {corr:.3f}")

# ============================================
# 6. OUTBREAK ANALYSIS
# ============================================

# count outbreaks by district
outbreak_counts = fever_df.groupby('district')['is_outbreak'].sum().sort_values(ascending=False)

print("\n" + "="*50)
print("OUTBREAK FREQUENCY BY DISTRICT")
print("="*50)
print(outbreak_counts.head(10))

# bar plot: outbreak frequency ----
fig, ax = plt.subplots(figsize=(10, 6))
outbreak_counts.head(10).plot(kind='barh', ax=ax, color='crimson')
ax.set_xlabel('Number of Outbreak Months', fontsize=11)
ax.set_ylabel('District', fontsize=11)
ax.set_title('Top 10 Districts by Outbreak Frequency', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('output/outbreak_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

# summary statistics by outbreak status ----
outbreak_summary = fever_df.groupby('is_outbreak')[['temp_avg', 'precipitation_avg', 'humidity_avg']].mean()

print("\n" + "="*50)
print("CLIMATE CONDITIONS: OUTBREAK vs NON-OUTBREAK")
print("="*50)
print(outbreak_summary)

print("\n✓ All visualizations saved to 'output/' folder")