import Marff as arff, newDcol, random, numpy
from math import sqrt
from sklearn.linear_model import perceptron


base=arff.abre_arff("/media/marcos/Data/Tese/Bags/1/IndividuoWine1.arff")
base2=arff.abre_arff("/media/marcos/Data/Tese/Bags/1/IndividuoWine2.arff")
X,y=arff.retorna_instacias(base)
X2,y2=arff.retorna_instacias(base2)
num_classes, todas_as_classes,*_=arff.retorna_classes_existentes(base)
print(todas_as_classes)

caminho="/media/marcos/Data/Tese/Bags/1/"
nome_base='IndividuoWine'


def fitness_andre(caminho, nome_base):
    dist_medias=list()
    complexidades=list()
    acuracia=list()
    perc = perceptron.Perceptron()
    for i in range(1,101):

        c=caminho +nome_base+ str(i) + ".arff"
        base = arff.abre_arff(c)
        X, y = arff.retorna_instacias(base)
        num_classes,*_=arff.retorna_classes_existentes(base)
        perc.fit(X, y)
        acuracia.append(perc.score(X, y))

        F1,N2,*_=newDcol.retorna_complexidade(c,complexidades="-F 1 -N 2", num_classes=num_classes, media=True)
        teste=[F1,N2]
        complexidades.append(teste)

    for j in range(len(complexidades)):
        dist=0
        for k in range(len(complexidades)):
            if(j==k):
                continue
            else:
                a=complexidades[j]
                b=complexidades [k]
                dist+=sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
        dist_medias.append(dist/100)

    print(acuracia)


def cruza(X,X2,y):
    inicio=0
    fim=0
    diferenca=0
    X3 = dict()
    X3['data'] = list()
    while(y[inicio]==y[fim] and diferenca<2):
        inicio=random.randint(0, len(y)-1)
        fim = random.randint(inicio, len(y) - 1)
        diferenca=fim-inicio
    for i in range(len(X)):
        if(i<inicio or i>fim):
            X3['data'].append(X[i])

        else:
            X3['data'].append(X2[i])
    for j in range(len(y)):
        X3['data'][j].append(y[j])

    #print(X3['data'])
    #
    x['data'] = [str(k) for k in X3]
    print(x['data'])
    return X3, x


    #print(X2)
    #print(X3)
   # print(len(X3))

print(base['attributes'])
x = [str(k) for k in todas_as_classes]
base['attributes'][-1] = ('Class', x)
print(base['attributes'])
#fitness_andre(caminho, nome_base)
dados,x=cruza(X,X2,y)

arff.cria_arff(base,x['data'],nome_base,'/home/marcos/Área\ de\ Trabalho/')