import csv as csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
file = open('train.csv')
file_csv = csv.reader(file)

# print(file_csv)

data =[]
for row in file_csv:
    data.append(row)

# print(pprint.pprint(data [2]))

header= data[0]
data.pop(0)

df= pd.DataFrame(data,columns= header)
# print(df)
# print(len(data))
# print(df.head())
# print(df.columns)


if __name__ == "__main__":
    print("Run as a script")


arrary = df.values
scaled_arrary= StandardScaler().fit_transform(arrary)
print(scaled_arrary[:3,:5])
