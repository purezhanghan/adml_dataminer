import pandas as pd
from copy import deepcopy
import numpy as np
from prudential_utils import paths
from sklearn import preprocessing
import scipy.stats as stats
from scipy.stats import norm

import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import operator
from ml_metrics import quadratic_weighted_kappa
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

print('Load the data using pandas')

path_train = paths.TRAIN_PATH
path_test = paths.TEST_PATH
path_submission = paths.SUBMISSION_PATH

def load_data_tree():
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    all_data = train.append(test)

    # fill test response with -1
    all_data['Response'].fillna(-1, inplace=True)
    all_data_non_response = deepcopy(all_data)
    all_data_non_response.drop(['Response'], axis=1, inplace=True)
    # feature engineering
    # 1.NaN values counting
    # 2.BMI age interaction
    # 3.Product_Info_2 factorize
    # 4.Medical History values summing up

    NA_count_list = []
    for row in range(len(all_data_non_response)):
        NA_count_list.append(all_data_non_response.iloc[row].isnull().sum())
    all_data['NA_count'] = NA_count_list

    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

    Employ_keyword_columns = all_data.columns[all_data.columns.str.startswith('Employment_Info_')]
    for employ in Employ_keyword_columns:
        all_data[employ].fillna(-1, inplace=True)


    Family_keyword_columns = all_data.columns[all_data.columns.str.startswith('Family_Hist_')]
    for family in Family_keyword_columns:
        all_data[family].fillna(all_data[family].mean(), inplace=True)

    all_data['Insurance_History_5'].fillna(all_data['Insurance_History_5'].mean(), inplace=True)
    all_data['Medical_History_1'].fillna(all_data['Medical_History_1'].median(), inplace=True)
    # important feature
    all_data['Medical_History_15'].fillna(float(0), inplace=True)
    all_data['Response'] = all_data['Response'].astype(int)

    Medical_history_columns = all_data.columns[all_data.columns.str.startswith('Medical_History_')]
    for medical in Medical_history_columns:
        all_data[medical].fillna(-1, inplace=True)

    binarizer = preprocessing.Binarizer()


    train_processed_temp = all_data[all_data['Response'] > 0].copy()
    target = train_processed_temp['Response']
    test_processed_tree = all_data[all_data['Response'] < 0].copy()
    test_processed_tree = test_processed_tree.drop('Response', axis=1)

    cols = list(train_processed_temp.columns.values)
    cols.remove('Id')
    corcoef = {}
    for col in cols:
        cor = np.corrcoef(train_processed_temp[col], train_processed_temp['Response'])
        corcoef[col] = cor[0, 1]
    sorted_cor = sorted(corcoef.items(), key=operator.itemgetter(1))

    # most 5 correlative
    positive_correlation = sorted_cor[-5:]

    variable_list = ['Medical_History_23', 'Product_Info_4', 'Medical_History_39','Medical_History_4','Medical_History_23']
    quad_list = [item+str('_quad') for item in variable_list]
    for i in range(len(variable_list)):
        all_data[quad_list[i]] = all_data[variable_list[i]] ** 2


    """
    polynomial feature particularly for medical_keyword
    -> 140 columns
    """

    poly_list = ['Medical_Keyword_15', 'Medical_Keyword_3', 'Med_Keywords_Count']
    poly = PolynomialFeatures(degree=3, interaction_only=True)
    poly.fit_transform(all_data[poly_list])

    all_data['Medical_Keyword_15_3'] = all_data['Medical_Keyword_15'] * all_data['Medical_Keyword_3']
    all_data['Medical_Keyword_3_Count'] = all_data['Medical_Keyword_3'] * all_data['Med_Keywords_Count']
    all_data['Medical_Keyword_15_Count'] = all_data['Medical_Keyword_15'] * all_data['Med_Keywords_Count']
    all_data['Medical_Keyword_3_15_Count'] = all_data['Medical_Keyword_3'] * all_data['Medical_Keyword_15'] * all_data['Med_Keywords_Count']
    all_data['Response'] = all_data['Response'].astype(int)

    train_processed_tree = all_data[all_data['Response'] > 0].copy()
    target = train_processed_tree['Response']
    test_processed_tree = all_data[all_data['Response'] < 0].copy()
    test_processed_tree = test_processed_tree.drop('Response', axis=1)

    return train_processed_tree, test_processed_tree

def data_normalization(train, test):

    all_data = train.append(test)
    all_data['Response'].fillna(-1, inplace=True)
    all_data_no_response = deepcopy(all_data)
    all_data_no_response.drop('Response', axis=1, inplace=True)

    # normalization

    cols_normalized = list(all_data_no_response.columns.values)
    cols_normalized.remove("Id")
    scalar = preprocessing.StandardScaler()
    all_data_no_response[cols_normalized] = scalar.fit_transform(all_data_no_response[cols_normalized])

    all_data_no_response['Response'] = all_data['Response'].astype(int)
    train_processed_linear = all_data_no_response[all_data_no_response['Response'] > 0].copy()
    test_processed_linear = all_data_no_response[all_data_no_response['Response'] < 0].copy()
    test_processed_linear = test_processed_linear.drop('Response', axis=1)

    return train_processed_linear, test_processed_linear


def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)


def eval_wrapper_xgb(preds, data):
    target = data.get_label()
    preds = np.array(preds)
    preds = np.clip(np.round(preds), np.min(target), np.max(target)).astype(int)
    return quadratic_weighted_kappa(preds, target)

def modelfit_xgb(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, metric='rmse', obt='reg:linear'):
    target = "Response"
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['objective'] = obt
        if xgb_param['objective'] == 'multi:softmax':
            xgb_param['num_class'] = 8
            metric = 'merror'
            xgtrain = xgb.DMatrix(dtrain[predictors].values, label=(dtrain[target]-1).values)
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics=metric, early_stopping_rounds=early_stopping_rounds, verbose_eval=3)
        alg.set_params(n_estimators=cvresult.shape[0])

    # cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
    #                   metrics='rmse', early_stopping_rounds=50, verbose_eval=3,
    #                   feval=eval_wrapper_xgb)

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric=metric)

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    if alg._estimator_type == 'regressor':
        dtrain_prediction = np.clip(dtrain_predictions,1,8)
        dtrain_predictions = np.round(dtrain_prediction).astype(int)
    # dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    # print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values,
                                                     dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.accuracy_score(dtrain['Disbursed'],
    #                                                dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # top 50 most important features
    feat_imp[:50].plot(kind="bar", title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    # return model which has optimal n_estimators for a specific learning_rate
    return alg



train_tree, test_tree = load_data_tree()

predictors = [x for x in train_tree.columns if x not in ["Id", "Response"]]


#### xgb1: classifier, xgb2: regressor

xgb1 = xgb.XGBClassifier(
    learning_rate = 0.1,
    n_estimators=1000,
    max_depth=7,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    nthread=4,
    scale_pos_weight=1,
    seed=27,

)

xgb2 = xgb.XGBRegressor(
    learning_rate = 0.1,
    n_estimators =1000,
    max_depth=7,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:linear',
    nthread=4,
    scale_pos_weight=1,
)


model = modelfit_xgb(xgb1, train_tree, predictors)
my_scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)


# tune max_depth and min_child_weight

param_test1 = {
    "max_depth":range(3,10,2),
    "min_child_weight":range(1,6,2)
}

params_test2 ={
    "max_depth":[4,5,6],
    "min_child_weight":[4,5,6]
}

# tune gamma
params_test3 = {
    "gamma" :[i/10.0 for i in range(0,5)]
}


# tune subsample and colsample_bytree
params_test4 = {
    'subsample' : [i/10.0 for i in range(6,10)],
    'colsample_bytree' : [i/10.0 for i in range(6,10)]
}

params_test5 = {
    'subsample' : [i/100.0 for i in range(75, 90, 5)],
    'colsample_bytree': [i/100.0 for i in range(75,90,5)]
}

# tune regularization parameters
params_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

params_test7 = {
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}

# tunning for learning_rate
xgb_for_learning_rate = deepcopy(model).set_params(n_estimators=5000, learning_rate=0.01)



def stage_one_grid_search_xgb(model, data, predictors, param_test):

    # if step == 2:
    #     param_test = param_test1
    #     print('\n grid search for max_depth in range[3, 10, 2] and min_child_weight in range[1, 6, 2]')
    # else:
    #     param_test = params_test2
    #     print('\n grid search for max_depth in [4,5,6] and min_child_weight in [4,5,6]')

    target = 'Response'
    # params = model.get_xgb_params()

    gsearch1  = GridSearchCV(estimator = model, param_grid = param_test,
                             scoring=my_scorer,  iid=False, cv=5,
                             verbose=3
    )

    gsearch1.fit(data[predictors], data[target])
    gsearch1.grid_scores_
    gsearch1.best_params_
    gsearch1.best_score_
    model.set_params(gsearch1.best_params_)

    return model











