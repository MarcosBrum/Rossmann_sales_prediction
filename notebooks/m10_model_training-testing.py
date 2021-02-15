import numpy as np
import pandas as pd
import random
import pickle
import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb

home_path = '/home/marcos/Documentos/comunidade_DS/DataScience_em_Producao/DataScience_Em_Producao/marcos/'
x_training = pd.read_pickle(home_path + 'ml_models_comp/x_training.pkl')
x_testing = pd.read_pickle(home_path + 'ml_models_comp/x_testing.pkl')

print(x_training.shape)

# ## 0.1 Helper Functions

def mean_percentage_error(y, yhat):
    return np.mean((y - yhat)/y)


def mean_absolute_percentage_error(y, yhat):
    return np.mean(np.abs((y - yhat)/y))


def ml_error(model_name, y, yhat):
    mae = mean_absolute_error(y, yhat)
    mape = mean_absolute_percentage_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))

    return pd.DataFrame({'Model Name': model_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}, index=[0])


# Used in section 7 - Machine Learning Modelling (cross validation)
def cross_validation(model, model_name, training_data=x_training, kfolds=5, verbose=False):
    mae_list = []
    mape_list = []
    rmse_list = []
    for k in reversed(range(1, kfolds+1)):
        # time intervals of 6 weeks
        validation_start_date = training_data['date'].max() - datetime.timedelta(days=k*6*7)
        validation_end_date = training_data['date'].max() - datetime.timedelta(days=(k-1)*6*7)

        # split data into training and validation
        training = training_data[training_data['date'] < validation_start_date]
        validation = training_data[(training_data['date'] >= validation_start_date) & (training_data['date'] <= validation_end_date)]

        xtraining = training.drop(['date', 'sales'], axis=1)
        ytraining = training['sales']

        xvalidation = validation.drop(['date', 'sales'], axis=1)
        yvalidation = validation['sales']

        # implement a model to get the results
        m = model.fit(xtraining, ytraining)
        yhat = m.predict(xvalidation)
        m_result = ml_error(model_name, np.expm1(yvalidation), np.expm1(yhat))

        mae_list.append(m_result['MAE'])
        mape_list.append(m_result['MAPE'])
        rmse_list.append(m_result['RMSE'])

    return pd.DataFrame({'Model name': model_name, 'MAE': np.round(np.mean(mae_list), 2).astype(str) + '+/-' + np.round(np.std(mae_list), 2).astype(str),
                         'MAPE': np.round(np.mean(mape_list), 2).astype(str) + '+/-' + np.round(np.std(mape_list), 2).astype(str),
                         'RMSE': np.round(np.mean(rmse_list), 2).astype(str) + '+/-' + np.round(np.std(rmse_list), 2).astype(str)}, index=[0])


# ## 8 Final Model

# # 8.1 Random Search

param = {'n_estimators': [1500, 1700, 2500, 3000, 3500],
         'eta': [0.01, 0.03],
         'max_depth': [3, 5, 9],
         'subsample': [0.1, 0.5, 0.7],
         'colsample_bytree': [0.3, 0.7, 0.9],
         'min_child_weight': [3, 8, 15]}

MAX_EVAL = 5

final_result = pd.DataFrame()
hyperparam_lst = []

for i in range(MAX_EVAL):
    # randomly selected hyperparameters
    hp = {k: random.sample(v, 1)[0] for k, v in param.items()}
    # print the selected ones
    print(hp)
    # store in list
    hyperparam_lst.append(hp.copy())
    # model
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                                 n_estimators=hp['n_estimators'],
                                 eta=hp['eta'],
                                 max_depth=hp['max_depth'],
                                 subsample=hp['subsample'],
                                 colsample_bytree=hp['colsample_bytree'],
                                 min_child_weight=hp['min_child_weight'])

    result = cross_validation(model=model_xgb, model_name='XGBoost Regressor')#, training_data=x_training, kfolds=5, verbose=False)
    iter_nr = pd.Series([i], name='Iter_nr')
    result_w_nr = pd.concat([result, iter_nr], axis=1)
    final_result = pd.concat([final_result, result_w_nr], ignore_index=True)

final_result.sort_values(by=['RMSE'], inplace=True, ignore_index=True)
final_result

# # 8.2 Final Model

final_result.loc[0, 'Iter_nr']

best_param_nr = final_result.loc[0, 'Iter_nr']

param_tuned = hyperparam_lst[best_param_nr]

x_train = x_training.drop(['date', 'sales'], axis=1)
y_train = x_training['sales']

x_test = x_testing.drop(['date', 'sales'], axis=1)
y_test = x_testing['sales']

# model
model_xgb_tuned = xgb.XGBRegressor(objective='reg:squarederror',
                                   n_estimators=param_tuned['n_estimators'],
                                   eta=param_tuned['eta'],
                                   max_depth=param_tuned['max_depth'],
                                   subsample=param_tuned['subsample'],
                                   colsample_bytree=param_tuned['colsample_bytree'],
                                   min_child_weight=param_tuned['min_child_weight']).fit(x_train, y_train)

yhat_xgb_tuned = model_xgb_tuned.predict(x_test)

xgb_result = ml_error('XGBoost Regressor', np.expm1(y_test), np.expm1(yhat_xgb_tuned))
xgb_result

mpe = mean_percentage_error(np.expm1(y_test), np.expm1(yhat_xgb_tuned))
mpe

pickle.dump(model_xgb_tuned, open(home_path + "ml_models_comp/model_xgb_tuned.pkl", "wb"), protocol=4)

pickle.dump(model_xgb_tuned, open(home_path + 'model/model_rossmann.pkl', 'wb'), protocol=4)

xgb_result.to_pickle(home_path + 'ml_models_comp/xgb_fine_tuned.pkl', protocol=4)

final_result.to_pickle(home_path + 'ml_models_comp/xgb_tunning_cv.pkl', protocol=4)
