import collections
from sklearn.externals.joblib import Parallel, delayed
import time, Cpx, Marff,csv,numpy
def parallel_distance2(i,bags,grupo, tipos):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx=(Cpx.complexity_data3(X_bag, y_bag,grupo,tipos))
   # _, score, _ = Cpx.biuld_classifier_over(X_bag, y_bag, =X_vali, y_vali,0.5)
    return cpx
def monta_arquivo(indx_bag):
    global X, y
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
    X_data = []
    y_data = []
    for i in indx_bag:
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    return X_data, y_data
nome_base = 'Adult'
local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
grupo=["overlapping","",'','','','']#ATECAO TESTEI PARA DUAS MEDIDAS DE DIFERENTES GRUPOS SOMENTE
#tipos=["F3",'N4','','','','']

arq_dataset = local_dataset + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
X, y, _ = Marff.retorna_instacias(arq_arff)
_, classes = Marff.retorna_classes_existentes(arq_arff)
tipos=None
for repeticao in range(1,2):
    #_, validation = Cpx.open_test_vali(local + "/", nome_base, repeticao)
   # X_vali, y_vali = Cpx.biuld_x_y(validation, X, y)
    bags = Cpx.open_bag(cpx_caminho + str(repeticao) + "/", nome_base)
    r = Parallel(n_jobs=-2)(delayed(parallel_distance2)(i, bags, grupo, tipos) for i in range(1,100))
print(r)

with open("xx.csv", 'w') as f:
    # print('entreivali')
    w = csv.writer(f)
    for i in r:
        w.writerow(i)