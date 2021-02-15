import pickle
import joblib


home_path = '/home/marcos/Documentos/comunidade_DS/DataScience_em_Producao/DataScience_Em_Producao/marcos/'

model_xgb_tuned = pickle.load(open(home_path + 'model/model_rossmann.pkl', 'rb'))

joblib.dump(model_xgb_tuned, open(home_path + "model/model_rossmann.joblib", "wb"), protocol=5)
