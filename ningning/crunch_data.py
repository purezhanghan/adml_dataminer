from load_data import train
import pandas as pd
import numpy as np

# Explore missing data
missing_data = train.isnull().sum()
# print(missing_data.dtype)
# print(type(missing_data))
total_data  = len(train)
df_missing_data = missing_data.to_frame()
df_missing_data.columns = ['counts']
# Identify missing categories
df_missing_data= df_missing_data[df_missing_data.counts != 0]
# Calculate missing percentage, we should delete the variables with 90% missing value
df_missing_data['missing_percent'] = df_missing_data.counts/total_data
print(df_missing_data)





