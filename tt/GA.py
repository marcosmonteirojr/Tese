import Marff as arff, newDcol, random, numpy
from math import sqrt
from sklearn.linear_model import perceptron


base=arff.abre_arff("/media/marcos/Data/Tese/Bags/1/IndividuoWine1.arff")
base2=arff.abre_arff("/media/marcos/Data/Tese/Bags/1/IndividuoWine99.arff")
X,y=arff.retorna_instacias(base)
X2,y2=arff.retorna_instacias(base2)
num_classes, todas_as_classes,*_=arff.retorna_classes_existentes(base)
#print(y)
#print(y2)

nome_base='Wine'
repeticao=1


def abre_arff(arquivo):
    '''funcao para abrir os caruivos arff de validacao e o bag'''
    global nome_base, repeticao
    c = "/media/marcos/Data/Tese/Bags/"+str(repeticao)+"/Individuo" + nome_base + str(arquivo) + ".arff"#arquivo bag
    v =  "/media/marcos/Data/Tese/Bases/Validacao/"+str(repeticao)+"/Valida" + nome_base +str(repeticao) + ".arff" # e um arquivo so de validacao por repeticao

    base = arff.abre_arff(c)
    base_valida = arff.abre_arff(v)

    X_val, y_val = arff.retorna_instacias(base_valida)
    X, y = arff.retorna_instacias(base)
    num_classes, *_ = arff.retorna_classes_existentes(base)


    return X_val,y_val, X, y, num_classes

def fitness_andre(caminho, valida_caminho, nome_base):

    perc = perceptron.Perceptron()

    acuracia = list()
    complexidades = list()
    for i in range(1,101):

        c=caminho +"/Individuo"+nome_base+str(i)+".arff"

        base = arff.abre_arff(c)
        base_valida=arff.abre_arff(valida_caminho)

        X_val, y_val=arff.retorna_instacias(base_valida)
        X, y = arff.retorna_instacias(base)
        num_classes,*_=arff.retorna_classes_existentes(base)

        perc.fit(X, y)
        acuracia.append(perc.score(X_val, y_val))

        F1,N2,*_=newDcol.retorna_complexidade(c,complexidades="-F 1 -N 2", num_classes=num_classes, media=True)
        teste=[F1,N2]
        complexidades.append(teste)

    dist_medias = list()
    for j in range(len(complexidades)):
        dist=0
        for l in range(len(complexidades)):

            if(j==l):
                continue
            else:
                a=complexidades[j]
                b=complexidades [l]
                dist+=sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
        dist_medias.append(dist/100)

    print(dist_medias)
    print(complexidades)
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


    print((len(X3['data'])))
    print(y)

    return X3


    #print(X2)
    #print(X3)
   # print(len(X3))

#print(base['attributes'])
f = [str(k) for k in todas_as_classes]
base['attributes'][-1] = ('Class', f)
#print(base['attributes'])
#print(X)
#x=cruza(X,X2,y)
#caminho = "/media/marcos/Data/Tese/Bags/"+str(1)

#arff.cria_arff(base,x,nome_base)


caminho = "/media/marcos/Data/Tese/Bags/"+str(1)
valida_caminho = "/media/marcos/Data/Tese/Bases/Validacao/"+str(1)+"/Valida" + nome_base +str(1) + ".arff"

fitness_andre(caminho, valida_caminho, nome_base)