from load_data import data
# import pandas as pd
# import numpy as np

# Explore missing data
missing_data = data.isnull().sum()
# print(missing_data.dtypes)
# print(type(missing_data))
total_data  = len(data)
df_missing_data = missing_data.to_frame()
df_missing_data.columns = ['counts']
# Identify missing categories
df_missing_data= df_missing_data[df_missing_data.counts != 0]
# Calculate missing percentage
df_missing_data['missing_percent'] = df_missing_data.counts/total_data
print(df_missing_data)


"""
# Reduce dimension: Identity highly irrelevant variables
# Id, Medical_History_10 (99.1% missing), Medical_History_24(93.6% missing), Medical_History_32 (98.1% missing)
# Remove target value 'Response'
train.drop(['Id','Response','Medical_History_10','Medical_History_24','Medical_History_32'], axis =1, inplace = True)
print(train.columns)

"""


class Missing:
    """
    This class will provide various method to handle missing values
    """
    def __init__(self, df):
        self.df = df









