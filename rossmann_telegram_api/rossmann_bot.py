import pandas as pd
import json
import requests

def load_dataset(store_id):
    home_path = '/home/marcos/Documentos/comunidade_DS/DataScience_em_Producao/DataScience_Em_Producao/marcos/'
    df10 = pd.read_csv(home_path + 'data/test.csv')
    df_store_raw = pd.read_csv(home_path + 'data/store.csv', low_memory=False)

    # merge test and store
    df_test = pd.merge(df10, df_store_raw, how='left', on='Store')

    df_test.sample(10)

    # choose stores
    df_test = df_test[df_test['Store'] == store_id]

    df_test = load_dataset()
    # remove closed days
    df_test = df_test[df_test['Open']!=0]
    df_test = df_test[~df_test['Open'].isnull()]
    df_test = df_test.drop('Id', axis=1)

    data = json.dumps(df_test.to_dict(orient='records'))

    return data

def predict_data(data):
    # API Call
    url = 'https://rossmann-predictions-marcos.herokuapp.com/rossmann/predict'
    header = {'Content-type': 'application/json'}
    data = data

    r = requests.post(url, data=data, headers=header)
    print('Status Code {}'.format(r.status_code))

    d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())
    return d1

# d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()
#
# for i in range(len(d2)):
#     print('Store number {} will sell R${:,.2f} in the next 6 weeks.'.format(d2.loc[i, 'store'], d2.loc[i, 'prediction']))
