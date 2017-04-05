

from ml_metrics import quadratic_weighted_kappa
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.cross_validation import  StratifiedKFold
from scipy.optimize import fmin_powell
from sklearn.metrics.scorer import make_scorer
from local import paths

# ensemble method for stacking five regression models
# stacking
# randomfores : bagging
# xgboost : boosting
# svm
# linear regression (l1 norm, l2 norm)
# 4 models to stack the genrealized model


linear_model = joblib.load('linear_model/linear.pkl')
rf_model =joblib.load('svm_model/xgb.pkl')
svm_model = joblib.load('rf_model/svm.pkl')
xgb_model = joblib.load('xgb_model/xgb.pkl')


np.random.seed(26)

train_tree = pd.read_csv(paths.TRAIN_TREE)
test_tree = pd.read_csv(paths.TEST_TREE)

train_regress = pd.read_csv(paths.TRAIN_REGRESS)
test_regress = pd.read_csv(paths.TEST_REGRESS)

# variable expect 'Id' and 'Response'
predictors = [col for col in train_regress.columns.values if col not in ['Id','Response']]

# customed evaluation function
myscorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)



num_classes = 8


# helper functions
def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)


def score_offset(data, bin_offset, sv, scorer=eval_wrapper):
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

def apply_offset(data, offsets):
    for j in range(num_classes):
        data[1, data[0].astype(int) == j] = data[0, data[0].astype(int)==j] + offsets[j]
    return data



def dataShuffle(train, test, shuffle=False):

    if shuffle:
        idx = np.random.permutation(train_tree['Response'].size)
        shuffledTrain = train.iloc[idx]
        shuffledTest= test.iloc[idx]

    return shuffledTrain, shuffledTest



n_folds = 5
skf = list(StratifiedKFold(y, n_folds))

preds_list = []
linear_model_predict_ensemble = np.zeros((X.shape[0], 1))

for i, (train, test) in enumerate(skf):

    print("Fold ", i)
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]

    linear_model.fit(X_train, y_train)

    train_preds = linear_model.predict(X_train)
    test_preds = linear_model.predict(X_test)

    train_preds = np.clip(train_preds, -0.99, 8.99)
    test_preds = np.clip(test_preds, -0.99, 8.99)

    offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])

    offset_preds = np.vstack((train_preds, train_preds, y_train))
    offset_preds = apply_offset(offset_preds, offsets)

    opt_order = list(range(8))
    for j in opt_order:
        train_offset = lambda x:-score_offset(offset_preds, x, j)*100
        offsets[j] = fmin_powell(train_offset, offsets[j], disp=True)

    test_offset = np.vstack((test_preds, test_preds))
    test_offset = apply_offset(test_offset, offsets)
    final_test_preds = np.round(np.clip(test_offset[1],1,8)).astype(int)

    preds_list.append(list(final_test_preds))
preds_result = [inner for outer in preds_list for inner in outer]
