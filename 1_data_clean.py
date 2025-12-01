# data cleaning and pre-processing

# load packages ----
import pandas as pd
import numpy as np

# load data ----
fever_df = pd.read_csv("data/raw/dengue_data.csv")

# inspect data ----
print(fever_df.shape) 
print(fever_df.head())              
print(fever_df.info())       
print(fever_df.columns.tolist())

# standardize column names ----
fever_df.columns = fever_df.columns.str.lower().str.replace(' ', '_', regex=True)

# add new cols for later ----