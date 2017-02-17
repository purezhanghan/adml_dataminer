from load_data import data
import pandas as pd
# import numpy as np


def check_missing(df):
    # Explore missing data
    missing_data = df.isnull().sum()
    # print(missing_data.dtypes)
    # print(type(missing_data))
    total_data = len(df)
    df_missing_data = missing_data.to_frame()
    df_missing_data.columns = ['counts']
    # Identify missing categories
    df_missing_data = df_missing_data[df_missing_data.counts != 0]
    # Calculate missing percentage
    df_missing_data['missing_percent'] = df_missing_data.counts / total_data
    print(df_missing_data)
    print(len(df_missing_data))
    # print(data.head())
    # print(data.dtypes)
    return df_missing_data

check_missing(data)


"""
# Reduce dimension: Identity highly irrelevant variables
# Id, Medical_History_10 (99.1% missing), Medical_History_24(93.6% missing), Medical_History_32 (98.1% missing)
# Remove target value 'Response'
train.drop(['Id','Response','Medical_History_10','Medical_History_24','Medical_History_32'], axis =1, inplace = True)
print(train.columns)

"""

# Create list of variable types

cont_variable_list = ['Product_Info_4', 'Ins_Age','Ht','Wt','BMI','Employment_Info_1','Employment_Info_4',
                      'Employment_Info_6','Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4',
                      'Family_Hist_5']


dis_variable_list = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24',
                     'Medical_History_32']
for i in range (48):
    i += 1
    dis_variable_list.append('Medical_Keyword_'+'i')


cat_variable_list =[]
for header in data.columns:
    if header in cont_variable_list and dis_variable_list:
        pass
    else:
        cat_variable_list.append(header)

missing_list = ['Employment_Info_1','Employment_Info_4','Employment_Info_6','Family_Hist_2','Family_Hist_3',
                'Family_Hist_4','Family_Hist_5','Insurance_History_5','Medical_History_1','Medical_History_10',
                'Medical_History_15','Medical_History_24','Medical_History_32']


class MissingMethod:
    """
    This class will provide various method to handle missing values
    """

    def __init__(self, df):
        self.df = df

    def fill_mode(self):
        for var in missing_list:
            if var in dis_variable_list:
                self.df[var] = self.df[var].fillna(self.df[var].mode()[0])
        return self.df

    def fill_avg(self):
        for var in missing_list:
            if var in cont_variable_list:
                self.df[var] = self.df[var].fillna(self.df[var].mean())
        return self.df






preprocess = MissingMethod(data).fill_mode()
preprocess = MissingMethod(data).fill_avg()
check_missing(preprocess)











