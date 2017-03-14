# train test split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
from sklearn.model_selection import train_test_split
df = pd.read_csv('cleaned_data1.csv',sep=',')
target = df['Response']

feature_col = df.columns[:-1]
X = df[feature_col]
Y = df['Response']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# model
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

model = LinearRegression()
model.fit(X, Y)
pred = model.predict(X)

from sklearn.linear_model import LogisticRegression
pred = pd.DataFrame(pred)
y_target = pd.DataFrame(Y.values).astype(int)
classifier = LogisticRegression ()
classifier.fit(pred, y_target)
prediction_result = classifier.predict(pred)

#pred = model.predict(X_test)
#pred = np.clip(pred, 1,8)
#prediction_result = np.round(pred)



plt.style.use('ggplot')

target_dis = Y.value_counts().sort_index()
pred_dis = pd.Series(prediction_result).value_counts().sort_index()

plt.figure(figsize=(20, 10))
df_plot = pd.concat([target_dis, pred_dis], axis=1)
df_plot.columns = ['target','pred']
df_plot.plot.bar()
plt.show()

