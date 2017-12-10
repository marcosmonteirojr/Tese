from math import sqrt
import Marff
from scipy.spatial import distance

base=Marff.abre_arff('/media/marcos/Data/Tese/Ag/1/IndividuoWine1.arff')
X,y=Marff.retorna_instacias(base)
num_classes,classes,*_=Marff.retorna_classes_existentes(base)
print(classes)

def retorna_range(X):
    '''
    retona listas com o maior, menor atributo e o range
    :@param X: array de instancias
    :@return: listRange, listMax, listMin
    '''
    listMax=[]
    listMin=[]
    listRange=[]
    cont=0

    while cont<len(X[0]):
        for i in range(len(X)):
            if(i==0):
                Min=X[i][cont]
                Max=X[i][cont]
                auxMin = 9999999
                auxMax = 0
            else:
                auxMin=X[i][cont]
                auxMax=X[i][cont]
            if(auxMin<Min):
                Min=auxMin
            if (auxMax > Max):
                Max = auxMax
        cont+=1
        listMax.append(Max)
        listMin.append(Min)
        listRange.append(Max-Min)
    #print(listMin, listMax, listRange)
    return listRange, listMax, listMin





def N2 (X, y, x_range):
    '''
     Retorna N2 para 2 classes

    :@param X: array de instancias
    :@param y: vetor de classes
    :@param x_range: array de range
    :@return: N2, intraClass, interClass
    '''
    inter = []
    intra = []
    for j in range(len(X)):
        min_intra = 99999
        min_inter = 99999
        for k in range(len(X)):
            if(j==k):
                continue
            else:
                a=X[j]
                b=X[k]
                dist=sqrt(sum(((a - b)/x_range) ** 2 for a, b, x_range in zip(a, b, x_range)))
                if(dist==0):
                    if y[j] != y[k]:
                        min_inter = dist
                       # min_intra=1
                    if y[j] == y[k]:
                        min_intra = dist
                        #min_inter=1
                if y[j]!=y[k] and dist<min_inter:
                    min_inter=dist
                if y[j]==y[k] and dist<min_intra:
                    min_intra=dist
        inter.append(min_inter)
        intra.append(min_intra)
    N2=sum(intra)/sum(inter)
    return N2, inter, intra


def Nn2 (X, y, x_range):
    '''
     Retorna N2 para 2 classes

    :@param X: array de instancias
    :@param y: vetor de classes
    :@param x_range: array de range
    :@return: N2, intraClass, interClass
    '''
    inter = []
    intra = []
    lista1 = []
    lista2 = []
    cont = 0

    for i in range(len(X)):
        if classes[cont] == y[i]:
            lista1.append(X[i])
        else:
            lista2.append(X[i])

    for j in range(len(lista1)):

        min_inter = 99999
        for k in range(len(lista2)):
            if(j==k):
                continue
            else:
                a=lista1[j]
                b=lista2[k]

                dist=sqrt(sum(((a - b)/x_range) ** 2 for a, b, x_range in zip(a, b, x_range)))
                if(dist==0):
                   min_inter = dist
                if dist<min_inter:
                    min_inter=dist
        inter.append(min_inter)
    for l in range(len(lista1)):
        min_intra = 99999
        for m in range(len(lista1)):
            if (l == m):
                continue
            else:
                a = lista1[l]
                b = lista1[m]
                dist = sqrt(sum(((a - b) / x_range) ** 2 for a, b, x_range in zip(a, b, x_range)))
                if (dist == 0):
                   # if y[j] != y[k]:
                        min_intra = dist

                if dist < min_intra:
                    min_intra = dist
        intra.append(min_intra)
    N2=sum(intra)/sum(inter)
    return N2, inter, intra





#for i in y:print(*lista1, *y, sep='\n')


x_range,*_=retorna_range(X)

kkkk,inter,intra=Nn2(X,y,x_range)

print(inter)
