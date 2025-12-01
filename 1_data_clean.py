# data cleaning and pre-processing

# load packages ----
import pandas as pd
import numpy as np
from scipy import stats

# load data ----
fever_df = pd.read_csv("data/raw/dengue_data.csv")

# inspect data ----
print(fever_df.shape) 
print(fever_df.head())              
print(fever_df.info())       
print(fever_df.columns.tolist())

# standardize column names ----
fever_df.columns = fever_df.columns.str.lower().str.replace(' ', '_', regex=True)

# outlier check ----
# z-score method for cases
fever_df['cases_zscore'] = stats.zscore(fever_df['cases'])
outliers = fever_df[np.abs(fever_df['cases_zscore']) > 3]
len(outliers)

# IQR method for climate variables
# flag outliers
for col in ['temp_avg', 'precipitation_avg', 'humidity_avg']:
    Q1 = fever_df[col].quantile(0.25)
    Q3 = fever_df[col].quantile(0.75)
    IQR = Q3 - Q1
    fever_df[f'{col}_outlier'] = ((fever_df[col] < (Q1 - 1.5 * IQR)) | 
                                   (fever_df[col] > (Q3 + 1.5 * IQR))).astype(int)
# sum outliers
for col in ['temp_avg', 'precipitation_avg', 'humidity_avg']:
    outlier_col = f'{col}_outlier'
    num_outliers = fever_df[outlier_col].sum()
    print(f'{col}: {num_outliers} outliers')

# add new cols for later ----
# date col
fever_df['date'] = pd.to_datetime(fever_df[['year', 'month']].assign(day=1))

# season col
# seasons pulled from Sri Lanka Met Department
# https://mausam.imd.gov.in/responsive/pdf_viewer_css/met1/Chapter-7%20page%20155-173/Chapter-7.pdf#:~:text=As%20per%20the%20Sri%20Lanka%20Met%20Department%2C,season%20(October%2DNovember)%20and%204)%20Northeast%20monsoon%20(December%2DFebruary).
def get_season(month):
    if month in [12, 1, 2]:
        return 'northeast_monsoon'  # dec. - feb.
    elif month in [3, 4]:
        return 'first_inter_monsoon'    # march - april
    elif month in [5, 6, 7, 8, 9]:
        return 'southwest_monsoon'  # may - sept.
    else:
        return 'second_inter_monsoon'    # oct. - nov.
    
fever_df['season'] = fever_df['month'].apply(get_season)

# log transform cases
fever_df['log_cases'] = np.log1p(fever_df['cases'])  # log1p handles zero cases

# temperature-rainfall interaction
fever_df['temp_precip_interaction'] = fever_df['temp_avg'] * fever_df['precipitation_avg']
    
# outbreak flag (cases > 75th percentile)
outbreak_threshold = fever_df['cases'].quantile(0.75)
fever_df['is_outbreak'] = (fever_df['cases'] > outbreak_threshold).astype(int)

# lagged cases: district (previous month for same district)
fever_df = fever_df.sort_values(['district', 'year', 'month'])
fever_df['cases_lag_district'] = fever_df.groupby('district')['cases'].shift(1)

# lagged cases: province (previous month for same province)
fever_df = fever_df.sort_values(['province', 'year', 'month'])
fever_df['cases_lag_province'] = fever_df.groupby('province')['cases'].shift(1)
    
# 3-month rolling average
fever_df['cases_3mo_avg'] = fever_df.groupby('district')['cases'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
    )

# quarter number
fever_df['quarter'] = fever_df['month'].apply(lambda x: (x-1)//3 + 1)

# monsoon flag
fever_df['is_monsoon'] = fever_df['season'].isin(['northeast_monsoon', 'southwest_monsoon']).astype(int)
    
# climate categories by percentiles
fever_df['temp_category'] = pd.qcut(fever_df['temp_avg'], 
                                    q=3, 
                                    labels=['Cool', 'Moderate', 'Hot'])
    
fever_df['rain_category'] = pd.qcut(fever_df['precipitation_avg'], 
                                    q=3, 
                                    labels=['Low', 'Medium', 'High'])

fever_df['humidity_category'] = pd.qcut(fever_df['humidity_avg'], 
                                        q=3, 
                                        labels=['Low', 'Medium', 'High'])

# save out ----
fever_df.to_csv("data/processed/dengue_data_cleaned.csv")