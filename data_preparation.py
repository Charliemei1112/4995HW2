import pandas as pd
import numpy as np

# load data
df = pd.read_csv('bank_marketing.csv')

# check for missing values
print(df.isnull().sum())

# check for duplicate values
print(df.duplicated().sum())

# check for outliers
print(df.describe())