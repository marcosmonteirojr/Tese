import pickle, Marff, Cpx, sys
import xgboost as xgb
from deslib.des import knora_e
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

nome_base = "Wine"

local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho = "/media/marcos/Data/Tese/Bases3/Bags/"
arq_dataset = local_dataset + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
X, y, _ = Marff.retorna_instacias(arq_arff, np_array=True)
X_train, y_train, X_test, y_test= train_test_split(X,y,test_size=0.25,train_size=0.5,stratify=y)
_, classes = Marff.retorna_classes_existentes(arq_arff)
resultado=[]

rng = np.random.RandomState(31337)
for repeticao in range(1,20):
    kf = KFold(n_splits=89, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X_train):
        xgb_model = xgb.XGBClassifier(n_estimators=100, n_jobs=6).fit(X[train_index], y[train_index])
        resultado.append(xgb_model.score(X[test_index],y[test_index]))

media=np.mean(resultado)
std=np.std(resultado)
arq1=open("XG_boost.csv","a")
arq1.write('{};{};{}\n'.format(nome_base, media, std))
arq1.close()
