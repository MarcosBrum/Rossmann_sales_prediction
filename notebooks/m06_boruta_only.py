import pandas as pd
import numpy as np

import datetime

from sklearn.ensemble import RandomForestRegressor

from boruta import BorutaPy

from IPython.display import display

"""# 6.0 Feature Selection"""

home_path = '/home/marcos/Documentos/comunidade_DS/DataScience_em_Producao/DataScience_Em_Producao/marcos/'
df6_1 = pd.read_csv(home_path + 'data_in_progress/df5_1.pkl')
df6_2 = pd.read_csv(home_path + 'data_in_progress/df5_2.pkl')
df6_3 = pd.read_csv(home_path + 'data_in_progress/df5_3.pkl')

df6 = pd.concat([df6_1, df6_1, df6_3], axis=0, ignore_index=True)

"""Remove repeated and auxiliary variables:"""

remove_cols = ['day_of_week', 'week_of_year', 'day', 'month', 'promo_since', 'competition_since', 'year_week']

df6 = df6.drop(remove_cols, axis=1)

df6['date'] = pd.to_datetime(df6['date'])

df6['sales'] = df6['sales'].astype(int)

df6['year'] = df6['year'].astype(int)

df6['competition_time_month'] = df6['competition_time_month'].astype(int)

"""## 6.1 Split original DataFrame into train and test"""

date_filter = df6['date'].max() - datetime.timedelta(days=6*7)

df6_train = df6[df6['date'] < date_filter]
df6_test = df6[df6['date'] >= date_filter]

X_train = df6_train.drop(['date', 'sales'], axis=1)
y_train = df6_train['sales']

X_test = df6_test.drop(['date', 'sales'], axis=1)
y_test = df6_test['sales']

"""## 6.2 Boruta as Feature selector"""

rf = RandomForestRegressor(n_jobs=-1)

"""The Boruta selector only takes arrays as input"""

X_train_arr = X_train.to_numpy()
y_train_arr = y_train.ravel()

boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42).fit(X_train_arr, y_train_arr)

cols_selected = boruta.support_.tolist()

#best features
cols_selected_boruta = X_train.iloc[:, cols_selected].columns.to_list()

#not selected
cols_not_selected_boruta = list(np.setdiff1d(X_train.columns, cols_selected_boruta))

display(cols_selected_boruta)

'''
Boruta ran 27 iterations in 1h10min, and it selected twelve features (fifteen were rejected):
cols_selected_boruta = ['store', 'promo', 'store_type', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2_since_week', 'promo2_since_year', 'promo_time_week', 'day_cos', 'day_of_week_sin', 'day_of_week_cos']
'''
