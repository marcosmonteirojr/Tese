from sklearn.ensemble import  BaggingClassifier
from sklearn.linear_model import perceptron as perc
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import Cpx, numpy as np
from deslib.dcs import LCA
import random
import Marff, sys

#base=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_sample.csv","r")
#lable=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_labels2.csv","r")

# base=open("/media/marcos/Data/Tese/Bases4/Dataset/features_tcnn_breakhis/features_dsfold1-40-train.txt","r")
# lable=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_labels.csv","r")
#
# #base=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_sample.csv","r")
# #lable=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_labels.csv","r")
# #random.seed(64)
# y=lable.readline()
#
# y=y[:-1].split(",")
# y=[float(elem) for elem in y]
# y=[int(elem) for elem in y]
# sample=[]
# #print(y)
# for i in base:
#     x=i[:-1].split(";")
#     x=[float(elem) for elem in x]
#    # print(x)
#     sample.append(x)
# print(sample)
# #exit(0)
# sample=np.array(sample)
# y=np.array(y)
# #print(y)
# scores=[]
#for i in range (20):
##    print(i)
#X_train, y_train, X_test, y_test, X_vali, y_vali, *_ = Cpx.split_data(sample,y)
#print(y_vali)
#print(y_train)


nome_base=sys.argv[1]
local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho = "/media/marcos/Data/Tese/Bases3/Bags/"
arq_dataset = local_dataset + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
X, y, _ = Marff.retorna_instacias(arq_arff)

teste, validation = Cpx.open_test_vali(local , nome_base, str(1))
X_vali, y_vali = Cpx.biuld_x_y(validation, X, y)
X_test, y_test = Cpx.biuld_x_y(teste,X,y)
treino=Cpx.open_training(local,nome_base,str(1))
#print(treino)
X_train, y_train = Cpx.biuld_x_y(treino,X,y)


clf= BaggingClassifier(perc.Perceptron(n_jobs=4,tol=1.0),n_estimators=100,max_samples=0.5)
clf.fit(X_train, y_train)

pool=clf.estimators_

def list_acert(pool, X_test, y_test):
    '''
    lista_acertos [classificador[instancias acertadas 1 ou nao 0]],lista com a soma de acerto por instanca (valores == 100 todos cla
    sificadores acertaram aquela instancia)
    :param pool: pool de classificadores
    :param X_test: X_teste
    :param y_test: y_teste
    :return: lista com os acertos [classificador][acertos ou erros], lista com a soma de acerto por instanca
    '''
    lista_acertos = np.zeros((100,len(y_test)), dtype=int)

    classificador_1 = []
    for j in range(len(pool)):
        x=pool[j].predict(X_test)
        for i in range(len(x)):
            if x[i]==y_test[i]:
                lista_acertos[j][i]=1
    lista_soma_acerto=np.sum(lista_acertos, axis=0)
    #print(lista_soma_acerto)
     #classificadores onde a instancia acertada somente 1 vez
    indice_inst_1, =np.where(lista_soma_acerto==1)#instancias acertadas somente 1 vez
    indice_inst_0, =np.where(lista_soma_acerto==0)

    if (indice_inst_1!=[]):
        for i in indice_inst_1:
            for j in range(len(lista_acertos)):
                if lista_acertos[j][i]==1:
                    classificador_1.append(j)
                    break
    else:
        print('N ha 1 acerto')

    return lista_acertos, indice_inst_0, indice_inst_1, classificador_1

lista_acertos, indice_inst_0, indice_inst_1, classificador_1=list_acert(pool,X_vali,y_vali)

#print((lista_acertos.tolist()))
nivel1=0
nivel2=0
nivel3=0
nivel4=0
nivel5=0
nivel6=0
nivel7=0
nivel100=0
nivel0=0
for i in lista_acertos.T:
    s=sum(i)
    if s == 0:
        nivel0 = nivel0 + 1
    if s>=1:
        nivel1=nivel1+1
    if s>=2:
        nivel2=nivel2+1
    if s>=3:
        nivel3=nivel3+1
    if s>=4:
        nivel4=nivel4+1
    if s>=5:
        nivel5=nivel5+1
    if s>=6:
        nivel6=nivel6+1
    if s>=7:
        nivel7=nivel7+1
    if s==len(X_vali):
        nivel100=nivel100+1


print(nivel0/len(X_vali),nivel1/len(X_vali),nivel2/len(X_vali),nivel3/len(X_vali),nivel4/len(X_vali),nivel5/len(X_vali),nivel6/len(X_vali),nivel7/len(X_vali),nivel100/len(X_vali))
txt=open("oracle.csv","a")
txt.write("{};{};{};{};{};{};{};{};{};{}\n".format(nome_base,nivel0/len(X_vali),nivel1/len(X_vali),nivel2/len(X_vali),nivel3/len(X_vali),nivel4/len(X_vali),nivel5/len(X_vali),
                                                   nivel6/len(X_vali),nivel7/len(X_vali),nivel100/len(X_vali)))
txt.close()

#    scores.append(clf_svm.score(X_test,y_test))
#    print(scores)
#
#print(np.average(scores))
#clf_perc = BaggingClassifier(base_estimator=Perceptron(n_jobs=4,tol=1.0),n_estimators=100, random_state=0, max_samples=0.5)
#clf_perc.fit(X_train, y_train)



#poolBag=clf_perc.estimators_


#print(poolBag)

#print(clf_svm.score(X_test,y_test))


#lcab=LCA(poolBag)
#lcab.fit(X_vali, y_vali)
#print(lcab.score(X_test,y_test))