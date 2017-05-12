import numpy as np
from copy import deepcopy
import pandas as pd



def proba_ensemble(rf_class_model, xgb_class_model, svm_class_model, regress_class_model, parsed_train, parsed_train_normal, parsed_test, parsed_test_normal):
	predictors = [col for col in parsed_train.columns.values if col not in ['Response', 'Id']]
	
	rf_class_model.fit(parsed_train[predictors], parsed_train['Response'])
	rf_class_model_train_preds_proba = rf_class_model.predict_proba(parsed_train[predictors])
	rf_class_model_test_preds_proba = rf_class_model.predict_proba(parsed_test[predictors])


	xgb_class_model.fit(parsed_train[predictors], parsed_train['Response'])
	xgb_class_model_train_preds_proba = xgb_class_model.predict_proba(parsed_train[predictors])
	xgb_class_model_test_preds_proba = xgb_class_model.predict_proba(parsed_test[predictors])

	svm_class_model.set_params(probability = True)
	svm_class_model.fit(parsed_train_normal[predictors], parsed_train_normal['Response'])
	svm_class_model_train_preds_proba = svm_class_model.predict_proba(parsed_train[predictors])
	svm_class_model_test_preds_proba = svm_class_model.predict_proba(parsed_test_normal[predictors])


	regress_class_model.fit(parsed_train_normal[predictors], parsed_train_normal['Response'])
	regress_class_model_train_preds_proba = regress_class_model.predict_proba(parsed_train_normal[predictors])
	regress_class_model_test_preds_proba = regress_class_model.predict_proba(parsed_test_normal[predictors])

	# class ensemble re-training
	ensemble_class_parsed_train = deepcopy(parsed_train)
	ensemble_class_parsed_test = deepcopy(parsed_test)

	class_proba_sum_up_train = rf_class_model_train_preds_proba + xgb_class_model_train_preds_proba + svm_class_model_train_preds_proba + regress_class_model_train_preds_proba
	proba_columns = ['class_one','class_two', 'class_three', 'class_four','class_five','class_six','class_seven','class_eight']
	class_proba_sum_up_train_df = pd.DataFrame(data=class_proba_sum_up_train, index=ensemble_class_parsed_train.index, columns=proba_columns)
	ensemble_class_parsed_train = pd.concat([parsed_train, class_proba_sum_up_train_df], axis=1)

	class_proba_sum_up_test = rf_class_model_test_preds_proba + xgb_class_model_test_preds_proba + svm_class_model_test_preds_proba + regress_class_model_test_preds_proba
	class_proba_sum_up_test_df = pd.DataFrame(data=class_proba_sum_up_test, index=ensemble_class_parsed_test.index, columns=proba_columns)
	ensemble_class_parsed_test = pd.concat([parsed_test, class_proba_sum_up_test_df], axis=1)

	new_predictors_class = [col for col in ensemble_class_parsed_train.columns.values if col not in ['Response','Id']]
	xgb_class_model.fit(ensemble_class_parsed_train[new_predictors_class], ensemble_class_parsed_train['Response'])
	ensemble_re_train_preds = xgb_class_model.predict(ensemble_class_parsed_test[new_predictors_class])
	ensemble_class_re_train_preds_round = np.clip(ensemble_re_train_preds.round(), 1, 8)

	return ensemble_class_re_train_preds_round
