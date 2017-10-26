#!/usr/bin/python
from __future__ import print_function
import sys
import numpy as np
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import os.path

#
arq = sys.argv[1]
#
f = open(arq, "r")
#
# interest = 1
# discard = 0
#
X = list()
Y = list()
Z = list()
Zerro = list()
#
if (os.path.exists("clf_svm.pkl") == False):
    #
    for i in f:
        tissue = int(i.split(";")[0].split("_")[0])
        if (tissue <= 4):
            class_t = 1
        else:
            class_t = 0
        Xj = list()
        for j in i[:-2].split(";")[2:]:
            Xj.append(float(j))
        if (len(Xj) == 162):
            X.append(Xj)
            Y.append(class_t)
            Z.append(i.split(";")[1])
        else:
            Zerro.append(i.split(";")[1])
    # for i in Zerro:
    #	print(i)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 6e-1, 4e-1, 2e-1],
                         'C': [5e-1, 5, 50, 500, 5000]},
                        {'kernel': ['linear'], 'C': [1e-1, 1, 10, 100, 1000]}]
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 6e-1, 4e-1, 2e-1],
    #            'C': [5e-1, 5, 50, 500, 5000]}]
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2e-1],
    #            'C': [500]}]
    #
    clf_svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='precision_weighted', n_jobs=4)
    clf_svm.fit(X, Y)
    #
    joblib.dump(clf_svm, "clf_svm.pkl")
else:
    clf_svm = joblib.load("clf_svm.pkl")
#
arq_64 = open("pftas_file_64.txt", "r")
arq_150 = open("pftas_file_150.txt", "r")
res_64 = open("res_file_64.txt", "w")
res_150 = open("res_file_150.txt", "w")
#
X_64 = list()
Y_64 = list()
Z_64 = list()
Zerro_64 = list()
#
for i in arq_64:
    class_str = i.split(";")[0]
    if (class_str == 'adenosis'):
        class_line = int(0)
    if (class_str == 'ductal_carcinoma'):
        class_line = int(1)
    if (class_str == 'fibroadenoma'):
        class_line = int(0)
    if (class_str == 'lobular_carcinoma'):
        class_line = int(1)
    if (class_str == 'mucinous_carcinoma'):
        class_line = int(1)
    if (class_str == 'papillary_carcinoma'):
        class_line = int(1)
    if (class_str == 'phyllodes_tumor'):
        class_line = int(0)
    if (class_str == 'tubular_adenoma'):
        class_line = int(0)

    Xj = list()
    for j in i[:-2].split(";")[2:]:
        Xj.append(float(j))
    if (len(Xj) == 162):
        X_64.append(np.array([Xj]))
        Z_64.append(i.split(";")[1])
    else:
        Zerro_64.append(i.split(";")[1])

for j in X_64:
    Y_64.append(clf_svm.predict_proba(j.reshape(1, -1)))
for j in range(len(X_64)):
    res_64.write("{};{}\n".format(Y_64[j], Z_64[j]))
#
X_150 = list()
Y_150 = list()
Z_150 = list()
Zerro_150 = list()
#
for i in arq_150:
    class_str = i.split(";")[0]
    if (class_str == 'adenosis'):
        class_line = int(0)
    if (class_str == 'ductal_carcinoma'):
        class_line = int(1)
    if (class_str == 'fibroadenoma'):
        class_line = int(0)
    if (class_str == 'lobular_carcinoma'):
        class_line = int(1)
    if (class_str == 'mucinous_carcinoma'):
        class_line = int(1)
    if (class_str == 'papillary_carcinoma'):
        class_line = int(1)
    if (class_str == 'phyllodes_tumor'):
        class_line = int(0)
    if (class_str == 'tubular_adenoma'):
        class_line = int(0)

    Xj = list()
    for j in i[:-2].split(";")[2:]:
        Xj.append(float(j))
    if (len(Xj) == 162):
        X_150.append(np.array([Xj]))
        Z_150.append(i.split(";")[1])
    else:
        Zerro_150.append(i.split(";")[1])

for j in X_150:
    Y_150.append(clf_svm.predict_proba(j.reshape(1, -1)))
for j in range(len(X_150)):
    res_150.write("{};{}\n".format(Y_150[j], Z_150[j]))
#
res_150.close()
res_64.close()
#
exit(0)


# adenosis;SOB_B_A-14-22549AB-40-014-448-0.png;0.12201;0.00953
import arff, os
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import perceptron
lables=[]
lables2=[]
dados = list()
treino = arff.load(open('/home/marcos/Documents/Tese/Distancias/TesteWine1.arff'))
teste = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))
nome_base="Wine"
#net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True)



if (os.path.exists("clf_pec"+nome_base+".pkl") == False):
# dados['data'] = list()
    perc = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True)
    for i in range (len(treino['data'])):
        lables.append(treino['data'][i][13])
    for i in treino['data']:
        dados.append(i[:-1])
    perc.fit(dados,lables)
    print(perc.score(dados,lables))
    joblib.dump(perc, "clf_pec"+nome_base+".pkl")
    print("entrei if")
else:
    perc = joblib.load("clf_pec"+nome_base+".pkl")
    for i in range (len(teste['data'])):
        dados2=np.array([dados[i]])
        print(perc.predict(dados2))
        print("else")