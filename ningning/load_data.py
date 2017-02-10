import pandas as pd

# Load data file
train = pd.read_csv('train.csv')
print(train.dtypes)
# print(train.Product_Info_2.str[0])

test = pd.read_csv('test.csv')
print(test.dtypes)





