import Marff, Cpx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import perceptron as perc
from deslib.dcs.ola import OLA
import numpy as np

#nome_base=sys.argv[1]
#bags_ga=sys.argv[2]
#print(bags_ga.split(','))
#exit(0)

nome_base='Banana'
#local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
bags_ga="20over2"
arq_dataset = caminho_base + "Dataset/" + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
#_,classes=Marff.retorna_classes_existentes(arq_arff)

X, y, data = Marff.retorna_instacias(arq_arff)


poolBag = []
poolPgsc = []

j=1


bags = Cpx.open_bag(cpx_caminho+str(j)+"/", nome_base)
bags2 = Cpx.open_bag(cpx_caminho+str(j)+"/", nome_base + bags_ga)
teste, validacao=Cpx.open_test_vali(local,nome_base,j)

X_test,y_test=Cpx.biuld_x_y(teste,X,y)
X_valida,y_valida=Cpx.biuld_x_y(validacao,X,y)
X_test=np.array(X_test)
X_valida=np.array(X_valida)
scaler = StandardScaler()
X_valida = scaler.fit_transform(X_valida)
X_test = scaler.transform(X_test)


def selecionados(select,n):
    """

    :param select: indices dos classificadores selecionados por instancia de teste
    :param n: numero de classificadores
    :return: O total de classificadores selecioandos
    """
    classifiers=[0]*n
    for i in select:
        classifiers[i] = classifiers[i] + 1
    return classifiers

for i in range(100):

    X_bag,y_bag=Cpx.biuld_x_y(bags['inst'][i],X,y)
    X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
    X_bag = scaler.transform(X_bag)
    X_bag2 = scaler.transform(X_bag2)

    percB = perc.Perceptron(n_jobs=4,max_iter=100,tol=10.0)
    percP = perc.Perceptron(n_jobs=4,max_iter=100,tol=10.0)

    poolBag.append(percB.fit(X_bag, y_bag))
    poolPgsc.append(percP.fit(X_bag2, y_bags2))


olab=OLA(poolBag)
olab.fit(X_valida, y_valida)
dist,f=olab._get_region_competence(X_test)
est=olab.estimate_competence(X_test,f,dist)
listtt=list(est)
print(listtt)
exit(0)
selected=olab.select(est)

olap=OLA(poolPgsc)
olap.fit(X_valida, y_valida)
dist2,f2=olap._get_region_competence(X_test)
est2=olap.estimate_competence(X_test,f2,dist2)#retorna a competencia de cada classificador, distancia da vizinhaca
selected2=olap.select(est2)#retorna uma lista com indice de cada classificador selecionado para cada instancia

bag =list(selected)
pgsc = list(selected2)
classifiers=selecionados(bag,100)
classifiers2=selecionados(pgsc,100)
print(selected)
print(classifiers)
print(classifiers2)





