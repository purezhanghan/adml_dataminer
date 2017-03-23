import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import svm
from sklearn.linear_model import BayesianRidge as BR
from scipy.optimize import fmin_powell
from local import paths
from local import utils

from ml_metrics import quadratic_weighted_kappa

# ensemble method for stacking five regression models

class ensemble_models():

    def __init__(self):
        return



clf_xgb = xgb.XGBRegressor(
            objective='reg:linear',
            learning_rate=0.05,
            min_child_weight=360,
            max_depth=7,
            n_estimators=700,
            seed = 2,
            subsample=0.85
)


clf_rf = RandomForestRegressor(
            n_estimators=1000,
            criterion='mse',
            max_depth=10,
            min_samples_split=10,
            max_features=0.8,
            max_leaf_nodes=None,
            bootstrap=True,
            random_state=2
)

clf_extra = ExtraTreesRegressor(
            n_estimators=1000,
            criterion='mse',
            max_depth=10,
            min_samples_split=10,
            max_features=0.62,
            min_weight_fraction_leaf=0.0,
            bootstrap=False,
            random_state=2

)

kernels = ['linear','poly','rbf','sigmoid','precomputed']

#
# for k in kernels:
#     clf_svm = svm.SVR(kernel=kernels)
#     clf_svm
