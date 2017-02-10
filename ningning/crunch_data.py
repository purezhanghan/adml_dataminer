# from load_data import df
# import pandas as pd
# import numpy as np
# print(df.columns)
# print(df.describe())
# print(df.dtypes)
# print(df.values) # object is not callable
# print(df['Id'])
# print(df.iloc[0:2,0:15]) # df.iloc[row, col]
# print(df.iloc[:,1:3])
# print(df.iat[2,2])
# print(pd.isnull(df))
# print(df['Employment_Info_1'])

# df['Product_Info_1'] = df['Product_Info_1'].astype('category')
# print(df['Product_Info_1'].dtypes)
# print(df['Product_Info_2'].dtypes)


# Transfer data type object to categorical
# def transfer_dtype(variable):
#     df[variable]= df[variable].astype('category')
#     print(df[variable].dtypes)


# transfer_dtype('Product_Info_1')
# transfer_dtype('Employment_Info_1',float) # not working for string
# transfer_dtype('Medical_History_1', int) # not working for int


# Transfer data type to continuous
# def transfer_dtype_continuous(variable):
#     df[variable]= pd.to_numeric(df[variable], errors='coerce')
#     print(df[variable].dtypes)


# Create list of variables
# cont_variable_list = ['Product_Info_4', 'Ins_Age','Ht','Wt','BMI','Employment_Info_1','Employment_Info_4',
#                       'Employment_Info_6','Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']

# dis_variable_list =['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']

# for i in range (48):
#     i += 1
#     print(i)
#     dis_variable_list.append('Medical_Keyword_'+'i')
# print(len(dis_variable_list))

# cat_variable_list =[]
# for header in df.columns:
#     if header in cont_variable_list and dis_variable_list:
#         pass
#     else:
#         cat_variable_list.append(header)


# Transfer to continuous data
# for var in cont_variable_list:
#     transfer_dtype_continuous(var)

# df['Medical_History_1']= pd.to_numeric(df['Medical_History_1'], errors='coerce').astype(int)
# print(df['Medical_History_1'].dtypes)
