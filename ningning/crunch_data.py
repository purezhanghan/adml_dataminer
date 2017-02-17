from load_data import data
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from functools import partial


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

# check_missing(data)


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
        self.df= df
        # self.df = df.drop('Response',axis =1, inplace = True)
        # print(df.shape)

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




# preprocess = MissingMethod(data).fill_mode()
# preprocess = MissingMethod(data).fill_avg()
# check_missing(preprocess)
# preprocess = MissingMethod(data)


# use SVD to fill missing values
def emsvd(df,k = None, tol = 1E-3, maxiter = None):
    if k is None:
        svdmethod = partial(np.linalg.svd, full_matrices = False)
    else:
        svdmethod = partial(svds, k =k)
    if maxiter is None:
        maxiter = np.inf

    col_avg = np.nanmean(df, axis=0, keepdims =1)
    valid = np.isfinite(df)
    y = np.where(valid, df, col_avg)

    halt = False
    ii =1
    v_prev =0.001

    while not halt:
        U, s, Vt = svdmethod(y - col_avg)
        y[~valid] =(U.dot(np.diag(s)).dot(Vt) + col_avg)[~valid]
        col_avg = y.mean(axis = 0, keepdims = 1)
        v = s.sum()
        if ii >= maxiter or ((v-v_prev)/v_prev) < tol:
            halt = True
        ii +=1
        v_prev = v

    return y

def fill_accuracy(initial, filled):
    diff = initial.sub(filled, axis =0)
    print(diff.head)
    # check_missing(diff)




data.drop('Response',axis =1, inplace =True)
# print(data.shape)
# print(data.index)
# print(type(data))
df = data.values
# print(df.shape)
# df_svd = emsvd(df)
# print(df_svd.shape)
# y = pd.DataFrame(df_svd, columns = data.columns)
# print(y.shape)

# y.to_csv('output.csv', sep =',')

# print(y.index)
# fill_accuracy(data,y)
# diff = data.subtract(y)

# print(diff.shape)

# diff = np.square(df-df_svd)
# print(diff)
# print(diff.shape)
# s = np.nansum(diff)
# np.set_printoptions(precision=1000)
# print(np.array(s))
# diff = pd.DataFrame(diff, columns = data.columns)
# diff.to_csv('diff.csv',sep = ',')


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

R = df
N = len(R)
M = len(R[0])
K =2

P  = np.random.rand(N,K)
Q = np.random.rand(M,K)
nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)

# print(nR)
















