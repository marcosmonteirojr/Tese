import Marff as arff, newDcol, random, numpy
from math import sqrt
from sklearn.linear_model import perceptron

'''
Variaveis globais

nome_base # tipo: string
          # conteudo: nome da base
          # utilizada em: abre_validacao, abre_individuos, retorna_complexidades
          # atribuida: (fora)
          # alterada por: nenhuma funcao

repeticao # tipo: int
          # conteudo: repeticao do sistema (1 a 20)
          # utilizada em: abre_validacao, abre_individuos, retorna_complexidades, cruza
          # atribuida: (fora)
          # alterada por: nenhuma funcao 

num_classes # tipo: int
            # conteudo: numero de classes da base em uso
            # utilizada em: retorna_complexidades
            # atribuida: abre_validacao
            # alterada por: nenhuma funcao 

X_val, y_val # tipo: lista de lista, lista
             # conteudo: instancias, classes
             # utilizada: fitness_andre
             # atribuida: abre_validacao
             # alterada por: nenhuma funcao 

dist_medias # tipo: lista
            # conteudo: distancia euclidiana media de 1 para N, (todos os bags)
            # utilizada: fitness_andre
            # atribuida: retorna_complexidades
            # alterada por: nenhuma funcao 


'''



def abre_validacao():

    global X_val, y_val, nome_base, repeticao, num_classes
    v = "/media/marcos/Data/Tese/Bases/Validacao/" + str(repeticao) + "/Valida" + nome_base + str(
        repeticao) + ".arff"  # e um arquivo so de validacao por repeticao
    base_valida = arff.abre_arff(v)

    X_val, y_val = arff.retorna_instacias(base_valida)
    num_classes,*_ = arff.retorna_classes_existentes(base_valida)
    #print(X_val)
    #return X_val,y_val

def abre_individuos(individuo):
    '''
    funcao para abrir os arquivos arff de validacao e o bag
    :param individuo: int correspondente ao valor do arff ex: IndividuoWine1
    :return: X, y, todas_as_classes, base(dataset) #
    '''
    global nome_base, repeticao, num_classes
    c = "/media/marcos/Data/Tese/Bags/"+str(repeticao)+"/Individuo" + nome_base + str(individuo) + ".arff"#arquivo bag
    base = arff.abre_arff(c)
    X, y = arff.retorna_instacias(base)
    _, todas_as_classes,*_ = arff.retorna_classes_existentes(base)

    return X, y, base, todas_as_classes

def retorna_complexidades():
    '''
    retorna a complexidade F1 e N2 baseado no nome da base e a repeticao, altera a variavel dist_media
    :param: num_classes # numero de classes e obtido pela fucao abre arquivos
    :return: complexidades #vetor de distancias medias e vetor de todas as complexiadades
    '''
    global nome_base, repeticao, num_classes, dist_medias
    complexidades = list()
    for i in range(1, 101):
        c = "/media/marcos/Data/Tese/Bags/" + str(repeticao) + "/Individuo" + nome_base + str(i) + ".arff"
        F1, N2, *_ = newDcol.retorna_complexidade(c, complexidades="-F 1 -N 2", num_classes=num_classes, media=True)
        cpx = [F1, N2]
        complexidades.append(cpx)

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

    return complexidades

def fitness_andre(individuo):
    '''
    Funcao de fitness, retorna a acuracia e a distancia do bag requerido
    :param individuo: tipo int #será passado por parametro para gerar o nome do arquivo
    :return: perc.score(X_val, y_val), dist_medias[arquivo]
    '''
    global dist_medias, X_val, y_val

    X, y,*_= abre_individuos(individuo)
    perc = perceptron.Perceptron()
    perc.fit(X, y)

    return perc.score(X_val, y_val), dist_medias[individuo]

def cruza(ind, ind2):
    global nome_base

    inicio=0
    fim=0
    diferenca=0
    X3=dict()
    X4 = dict()
    X3['data']= list()
    X4['data'] = list()
    X,y,base, todas_as_classes=abre_individuos(ind)
    X2,y2,*_=abre_individuos(ind2)

    while(y[inicio]==y[fim]):
        inicio=random.randint(0, len(y)-1)
        fim = random.randint(inicio, len(y) - 1)
        #diferenca=fim-inicio
    for i in range(len(X)):
        if(i<=inicio or i>=fim):
            X3['data'].append(X[i])
            X4['data'].append(X2[i])
        else:
            X3['data'].append(X2[i])
            X4['data'].append(X[i])

    for j in range(len(y)):
        X3['data'][j].append(y[j])
        X4['data'][j].append(y[j])
    #rep=repeticao+1

    pasta=""
    nome=nome_base+str(1)
    nome2 = nome_base + str(2)
    arff.cria_arff(base,X3,todas_as_classes,pasta,nome)
    arff.cria_arff(base,X4,todas_as_classes,pasta,nome2)
    print(((X is  X2)))
    print(((inicio,fim)))
    print(((X3['data'][:][-1])))
   # print((len(X4['data'])))
    print((y[inicio],y[fim]))

    #return X3


    #print(X2)
    #print(X3)
   # print(len(X3))

#print(base['attributes'])
#f = [str(k) for k in todas_as_classes]
#base['attributes'][-1] = ('Class', f)
#print(base['attributes'])
#print(X)
#x=cruza(X,X2,y)
#caminho = "/media/marcos/Data/Tese/Bags/"+str(1)

#arff.cria_arff(base,x,nome_base)
nome_base='Wine'
repeticao=1
num_classes=2
dist_medias=[]
X_val=[]
y_val=[]


abre_validacao()
_=retorna_complexidades()
acur, dist=fitness_andre(2)
cruza(1,2)
#print(acur, dist)
#fitness_andre(caminho, valida_caminho, nome_base)




















#################################################################################
#funcao sem uso

# def fitness_andre(caminho, valida_caminho, nome_base):
#
#     perc = perceptron.Perceptron()
#
#     acuracia = list()
#     complexidades = list()
#     for i in range(1,101):
#
#         c=caminho +"/Individuo"+nome_base+str(i)+".arff"
#
#         base = arff.abre_arff(c)
#         base_valida=arff.abre_arff(valida_caminho)
#
#         X_val, y_val=arff.retorna_instacias(base_valida)
#         X, y = arff.retorna_instacias(base)
#         num_classes,*_=arff.retorna_classes_existentes(base)
#
#         perc.fit(X, y)
#         acuracia.append(perc.score(X_val, y_val))
#
#         F1,N2,*_=newDcol.retorna_complexidade(c,complexidades="-F 1 -N 2", num_classes=num_classes, media=True)
#         teste=[F1,N2]
#         complexidades.append(teste)
#
#     dist_medias = list()
#     for j in range(len(complexidades)):
#         dist=0
#         for l in range(len(complexidades)):
#
#             if(j==l):
#                 continue
#             else:
#                 a=complexidades[j]
#                 b=complexidades [l]
#                 dist+=sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
#         dist_medias.append(dist/100)
#
#     print(dist_medias)
#     print(complexidades)
#     print(acuracia)