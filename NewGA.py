import Marff as arff, newDcol, random, os, shutil, sys
from math import sqrt
from sklearn.linear_model import perceptron
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

def retorna_complexidades(population=None, primeira=None):
    '''
    retorna a complexidade F1 e N2 baseado no nome da base e a repeticao, altera a variavel dist
    possui tres situacoes, 1 primeira geracao, demais populacao (nova populacao, antes da geracao), e populacao geracao (populacao
    de uma nova geracao)

    #################333continuar #####################3333
    :param: populacao, e primeira geracao
    :return: complexidades #vetor de distancias medias e vetor de todas as complexiadades, e variavel global dist

    '''

    global nome_base, repeticao, num_classes, geracao, caminho_todas, pop, contador_complexidades, nome_individuo, dist
    complexidades = list()

   # print('##########################-Complexidades-##############################################')
    if (geracao == 0 and primeira==True):
        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        dist['nome'] = pop
        #print('primeira populacao', dist['nome'])
        for i in dist['nome']:
            indx = abre_arquivo(i)
            X,y=monta_arquivo(indx)
            # calcula F1 e N2  F1, N2, =
            F1,N2=0################################3
            cpx = [F1, N2]
            complexidades.append(cpx)
        for j in range(len(complexidades)):
            dista = 0
            for l in range(len(complexidades)):
                if (j == l):
                    continue
                else:
                    a = complexidades[j]
                    b = complexidades[l]
                    dista += sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
            dist['dist'].append(dista / 100)
        #print('complexidade', dist['dist'])
        return complexidades

    if (population):
        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        dist['nome'] = population
       # print('geracao de populacao', dist['nome'])
        for i in dist['nome']:
            # print(i[0])
            c = caminho_todas + str(repeticao) + "/" + str(geracao) + "/Individuo" + nome_base + str(i[0]) + ".arff"
            F1, N2, *_ = newDcol.retorna_complexidade(c, complexidades="-F 1 -N 2", num_classes=num_classes, media=False)
            cpx = [F1, N2]
            # print(cpx)
            complexidades.append(cpx)
        for j in range(len(complexidades)):
            dista = 0
            for l in range(len(complexidades)):
                if (j == l):
                    continue
                else:
                    a = complexidades[j]
                    b = complexidades[l]
                    dista += sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
                dist['dist'].append(dista / 100)
       # print('complexidade', dist['dist'])
    else:

        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        inicio=nome_individuo-100
        for i in range(inicio,nome_individuo):
            x=[]
            x.append(i)
            dist['nome'].append(x)
        #print('demais populacao', dist['nome'])
        complexidades = list()
        for i in dist['nome']:#
            indx = abre_arquivo(i)
            X, y = monta_arquivo(indx)
            # calcula F1 e N2  F1, N2, =
            F1, N2 = 0  ################################3
            cpx = [F1, N2]
            complexidades.append(cpx)
        for j in range(len(complexidades)):
            dista = 0
            for l in range(len(complexidades)):
                if (j == l):
                    continue
                else:
                    a = complexidades[j]
                    b = complexidades[l]
                    dista += sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
            dist['dist'].append(dista / 100)
        #print('complexidade', dist['dist'])
    return complexidades
def altera_arquivo_marcelo():
    '''
    da nome aos bags nesse caso 1 a 100
    :return:
    '''
    arq = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + ".indx")
    arqtemp = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + ".indxTemp", 'w')
    cont=1
    for i in arq:
        texto=i
        q=texto.split(" ")
        print(q)
        q.insert(0,str(cont))
        #print(q)
        for j in q:
           # print(j)
            if(j!=q[-1]):
                arqtemp.write(j)
                arqtemp.write(" ")
            else:
                arqtemp.write(j)
                #arqtemp.write('\n')
        cont+=1
    arq.close()
    arqtemp.close()
    os.system("cp -r /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + ".indxTemp /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + ".indx")
    os.system("rm /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + ".indxTemp")



def abre_arquivo(individuo):
    global nome_base, repeticao

    arq=open("/media/marcos/Data/Tese/GA2/"+str(repeticao)+"/"+nome_base+".indx")
    for i in arq:
        texto=i
        #print(str(individuo))
        if(str(individuo)==texto.split(" ")[0]):
            indx_bag=texto.split(" ")
            arq.close()
            break

    return indx_bag[1:]

def monta_arquivo(indx_bag):
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :return:
    '''
    print(indx_bag)
    global nome_base
    X_data=[]
    y_data=[]
    arq2=("/media/marcos/Data/Tese/Bases2/Dataset/"+nome_base+".arff")
    arq3=arff.abre_arff(arq2)
    X,y=arff.retorna_instacias(arq3)
    for i in indx_bag:
        #print(int(i))
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    print(X_data)
    exit(0)
    return X_data, y_data

def cruza(ind1, ind2):
    '''
    Para funcionar os bags devem ter o mesmo tamanho
    :param ind1:
    :param ind2:
    :return:
    '''
    global nome_individuo
    inicio=fim=0
    ind_out1=[]
    
    indx_bag1=abre_arquivo(ind1[0])
    indx_bag2=abre_arquivo(ind2[0])
    X, y_data=monta_arquivo(indx_bag2)
    while (y_data[inicio] == y_data[fim]):
        inicio = random.randint(0, len(y_data) - 1)
        fim = random.randint(inicio, len(y_data) - 1)
    for i in range(len(X)):
        if (i <= inicio or i >= fim):
            ind_out1.append(indx_bag1[i])
        else:
            ind_out1.append(indx_bag2[i])
    print(y_data[inicio],y_data[fim])
    print(indx_bag1)
    print(indx_bag2)
    print(ind_out1)




def mutacao(ind):
    global geracao, off, nome_individuo



    indx_bag1 = abre_arquivo(ind[0])
    X,y_data=monta_arquivo(indx_bag1)
    ind_out=[]
    inst = 0
    inst2 = len(y_data)

    if (geracao == 0 and off == []):
        ind2 = random.randint(1, 100)
    else:
        ind2 = random.sample(off, 1)

        # print('mutacaooooo e contador', individuo, ind2, contador_cruzamento)
    indx_bag2 = abre_arquivo(ind2)
    X2, y2_data = monta_arquivo(indx_bag2)

    while y_data[inst] != y2_data[inst2 - 1]:
        inst = random.randint(0, len(y_data) - 1)
        inst2 = random.randint(0, len(y2_data) - 1)
    for i in range(len(indx_bag1)):
        if(i==inst):
            ind_out.append(indx_bag2[i])
        else:
            ind_out.append(indx_bag1[i])




def fitness_moga(individuo):

    '''
    Funcao de fitness, retorna a acuracia e a distancia do bag requerido
    :param individuo: tipo int #será passado por parametro para gerar o nome do arquivo
    :return: perc.score(X_val, y_val), dist_medias[arquivo]
    '''
    global dist, X_val, y_val
    # print('fitness')
    distancia = 0

    ind = individuo[0]
    for i in range(len(dist['nome'])):

        if dist['nome'][i][0] == ind:
            distancia = dist['dist'][i]
            # print('nome, distancia', dist['nome'][i][0], distancia)
            break
    X, y, *_ = abre_individuos(ind)
    perc = perceptron.Perceptron()
    perc.fit(X, y)
    # print(len(dist_medias))

    out = float(perc.score(X_val, y_val))
    out2 = float(distancia)
    return out, out2,
repeticao=1
off=[]
geracao=0
nome_base="Wine"
#altera_arquivo_marcelo()
cruza([1],[99])
#nome_bag()
#print(x)

