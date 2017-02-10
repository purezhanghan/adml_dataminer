import pandas as pd

# Load data file
# train = pd.read_csv('train.csv')
# print(train.dtypes)
# print(train.Product_Info_2.str[0])

# test = pd.read_csv('test.csv')
# print(test.dtypes)

def data_loader(path):
    file = pd.read_csv(path)
    return file

train= data_loader('train.csv')
# print(train)
test= data_loader('test.csv')
# print(test)



