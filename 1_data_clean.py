# data cleaning and pre-processing

# load packages ----
import pandas as pd
import numpy as np
from scipy import stats

# load data ----
fever_df = pd.read_csv("data/raw/Dengue Hotspot Data.csv")

# inspect data ----
print(fever_df.shape) 
print(fever_df.head())              
print(fever_df.info())       
print(fever_df.columns.tolist())

# check missing values ----
print(fever_df.isna().sum())

# standardize column names ----
fever_df.columns = fever_df.columns.str.lower().str.replace(' ', '_', regex=True)

# outlier check ----
# z-score method for cases
fever_df['cases_zscore'] = stats.zscore(fever_df['total_cases'])
outliers = fever_df[np.abs(fever_df['cases_zscore']) > 3]
len(outliers)

# IQR method for climate variables
# flag outliers
for col in ['current_temperature', 'current_precipitation', 'current_specific_humidity']:
    Q1 = fever_df[col].quantile(0.25)
    Q3 = fever_df[col].quantile(0.75)
    IQR = Q3 - Q1
    fever_df[f'{col}_outlier'] = ((fever_df[col] < (Q1 - 1.5 * IQR)) | 
                                   (fever_df[col] > (Q3 + 1.5 * IQR))).astype(int)
# sum outliers
for col in ['current_temperature', 'current_precipitation', 'current_specific_humidity']:
    outlier_col = f'{col}_outlier'
    num_outliers = fever_df[outlier_col].sum()
    print(f'{col}: {num_outliers} outliers')

# add new cols for later ----
# date col
fever_df['date'] = pd.to_datetime(fever_df[['year', 'month' ,'day']])

# season col
def get_season(month):
    if month in [12, 1, 2, 3]:
        return 'rainy_season'  # dec. - march.
    else:
        return 'dry_season'    # april - nov.
    
fever_df['season'] = fever_df['month'].apply(get_season)

# log transform cases
fever_df['log_cases'] = np.log1p(fever_df['total_cases'])  # log1p handles zero cases

# outbreak flag (cases > 75th percentile)
outbreak_threshold = fever_df['total_cases'].quantile(0.75)
fever_df['is_outbreak'] = (fever_df['total_cases'] > outbreak_threshold).astype(int)

# lagged cases (previous month for same city)
fever_df = fever_df.sort_values(['city', 'year', 'month'])
fever_df['cases_lag'] = fever_df.groupby('city')['total_cases'].shift(1)
    
# 3-month rolling average
fever_df['cases_3mo_avg'] = fever_df.groupby('city')['total_cases'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
    )

# quarter number
fever_df['quarter'] = fever_df['month'].apply(lambda x: (x-1)//3 + 1)

# climate categories by percentiles
fever_df['temp_category'] = pd.qcut(fever_df['current_temperature'], 
                                    q=3, 
                                    labels=['Cool', 'Moderate', 'Hot'])
    
fever_df['rain_category'] = pd.qcut(fever_df['current_precipitation'], 
                                    q=3, 
                                    labels=['Low', 'Medium', 'High'])

fever_df['humidity_category'] = pd.qcut(fever_df['current_specific_humidity'], 
                                        q=3, 
                                        labels=['Low', 'Medium', 'High'])

# save out ----
fever_df.to_csv("data/processed/dengue_data_cleaned.csv", index=False)