import Marff
import multiprocessing
import random, os
from sklearn.linear_model import perceptron
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.preprocessing import MinMaxScaler
import complexity_pcol as dcol
import numpy as np
from math import sqrt
from scoop import futures

def altera_arquivo_marcelo():
    '''
    da nome aos bags nesse caso 1 a 100
    :return: altera os arquivos
    '''
    global repeticao, nome_base, geracao, caminho_bags, caminho_originais
    arq = open(caminho_originais + "Originais"+str(repeticao) + "/" + nome_base+".idx")
    arqtemp = open(caminho_originais + "Originais"+str(repeticao) + "/" + nome_base +str(geracao)+ ".indxTemp", 'w')
    cont=1
    for i in arq:
        texto=i
        q=texto.split(" ")
        #print(q)
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
    os.system("cp -r "+caminho_originais + "Originais"+str(repeticao) + "/" + nome_base + str(geracao)+ ".indxTemp "+caminho_bags + str(repeticao) + "/" + nome_base +str(geracao)+ ".indx")
    os.system("rm "+caminho_originais + "Originais"+str(repeticao) + "/" + nome_base + str(geracao)+ ".indxTemp")

def distancia(primeira=False,population=None):
    global classes, off, pop, nome_individuo,dist, geracao, dispersao
    if(geracao==30 and dispersao==True):
        arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + "-3.indx")
    else:
        arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indx")
   # print(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indx")
    texto = arq.readlines()
    if (primeira and geracao==0):
        compx = []
        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        print("primeira")
        dist['nome'] = pop
        for i in range(0,100):
            text = texto[i]
            indx_bag = text.split(" ")
            indx_bag = indx_bag[1:]
           # print(indx_bag)
            X_29, y_29 = monta_arquivo(indx_bag, True)
            scaler = MinMaxScaler()
            scaler.fit(X_29)
            transformed_data = scaler.transform(X_29)
            complex = dcol.PPcol(classes=classes)
            complexidades = complex.xy_measures(transformed_data, y_29)
            F1=(np.average(complexidades['F1']))
            N2=(np.average(complexidades['N2']))
            cpx=[F1,N2]
            compx.append(cpx)
        #print(compx)
        for j in range(len(compx)):
            dista = 0
            for l in range(len(compx)):
                if (j == l):
                    continue
                else:
                    a = compx[j]
                    b = compx[l]
                    dista += sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
                   # print(j,l)
            dist['dist'].append(dista / 100)
       # print('tamanho das dist inicial', len(dist['dist']))

    if (population!=None):
        print("populacao")
        compx = []
        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        dist['nome'] = population
        for i in dist['nome']:
            #print(i[0])
            for j in texto:
               # print(j.split(" ")[0])
                if (str(i[0]) == j.split(" ")[0]):
                    indx_bag=j.split(" ")
                    indx_bag = indx_bag[1:]
                   # print("teste", str(i[0]), j.split(" ")[0])
         #           print(indx_bag)

                    X_29, y_29 = monta_arquivo(indx_bag, True)

                    scaler = MinMaxScaler()
                    scaler.fit(X_29)
                    transformed_data = scaler.transform(X_29)
                    complex = dcol.PPcol(classes=classes)
                    complexidades = complex.xy_measures(transformed_data, y_29)
                    F1 = (np.average(complexidades['F1']))
                    N2 = (np.average(complexidades['N2']))
                    cpx = [F1, N2]
                    compx.append(cpx)
       # print('tamanho das compxlation popu', len(compx))
        for k in range(len(compx)):
            dista = 0
            for l in range(len(compx)):
                if (k == l):
                    continue
                else:
                    a = compx[k]
                    b = compx[l]
                    dista += sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
                    # print(j,l)
            dist['dist'].append(dista / len(population))

    if(primeira==False and population==None):
        compx = []
        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        inicio = nome_individuo - numero_individuo
        #print(inicio)
        for i in range(inicio, nome_individuo):
           # print(i)
            x = []
            x.append(i)
            dist['nome'].append(x)
        for j in range(100,numero_individuo+100):
            # print(j.split(" ")[0])
            #print(j)

            text = texto[j]
            indx_bag = text.split(" ")
            indx_bag = indx_bag[1:]
           # print(indx_bag[0])

            X_29, y_29 = monta_arquivo(indx_bag, True)
           # print("teste")
            scaler = MinMaxScaler()
            scaler.fit(X_29)
            transformed_data = scaler.transform(X_29)
            complex = dcol.PPcol(classes=classes)
            complexidades = complex.xy_measures(transformed_data, y_29)
            F1 = (np.average(complexidades['F1']))
            N2 = (np.average(complexidades['N2']))
            cpx = [F1, N2]
            compx.append(cpx)

       # print('tamanho das compxlation outras', len(compx))
        for k in range(len(compx)):
            dista = 0
            for l in range(len(compx)):
                if (k == l):
                    continue
                else:
                    a = compx[k]
                    b = compx[l]
                    dista += sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
                    # print(j,l)
            dist['dist'].append(dista / numero_individuo)
    arq.close()
    #compx.clear()
    return dist

def abre_arquivo(individuo=None, valida=False):
    '''

    :param individuo: nome do individuo
    :param valida: para aproveitar a funcao ela abre a validacao
    :return: linha com indices do arquivo, caso do individuo exclui a primeira coluna, validacao nao
    '''
    global nome_base, repeticao, geracao, caminho_bags, caminho_base
    if individuo:
        arq=open(caminho_bags+str(repeticao)+"/"+nome_base+str(geracao)+".indx")
        #print(arq)
        for i in arq:
            texto=i
            #print('abre arquivo indi',str(individuo))
            if(str(individuo)==texto.split(" ")[0]):
                indx_bag=texto.split(" ")
                arq.close()
                indx_bag=indx_bag[1:]
                break

    elif valida:
        arq = open(caminho_base+"Validacao/" + str(repeticao) + "/" + nome_base+".idx")
        texto=arq.readline()
        indx_bag=texto.split(" ")
        arq.close()

    #print(indx_bag)
    return indx_bag

def monta_arquivo(indx_bag,vet_class=False):
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
    global nome_base, classes, caminho_base
    X_data=[]
    y_data=[]
    arq2=(caminho_base+"Dataset/"+nome_base+".arff")
    arq3=Marff.abre_arff(arq2)
    X,y,_=Marff.retorna_instacias(arq3)
    if(vet_class):
        _,classes,_,_=Marff.retorna_classes_existentes(arq3)
    for i in indx_bag:
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    return X_data, y_data

def cruza(ind1, ind2):
    '''
    Para funcionar os bags devem ter o mesmo tamanho
    :param ind1:
    :param ind2:
    :return:
    '''
    global nome_individuo, repeticao, nome_base, geracao, caminho_bags, dispersao, contador_cruzamento, numero_individuo
    #print("Cruzamento")
    #print(ind1, ind2, geracao)
    individuo_arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao)+".indx", 'a')
    inicio=fim=0
    ind_out1=[]
    ind_out1.append(str(nome_individuo))
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
    ind1[0] = nome_individuo
    ind2[0] = nome_individuo
    nome_individuo+=1
    for j in ind_out1:
        if (j != ind_out1[-1]):
            individuo_arq.write(j)
            individuo_arq.write(" ")
        else:
            individuo_arq.write(j)

    if (dispersao == True):

        contador_cruzamento = contador_cruzamento + 1
        if (contador_cruzamento == numero_individuo+1):
            individuo_arq.close()
            contador_cruzamento = 1
            distancia(primeira=False,population=None)
    return creator.Individual(ind1), creator.Individual(ind2)



def mutacao(ind):

    global geracao, off, nome_individuo, repeticao, caminho_bags, dispersao, contador_cruzamento, numero_individuo
   # print("mutacao")
    #print("off", (off))
    individuo_arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indx",
                         'a')
    indx_bag1 = abre_arquivo(ind[0])
    X,y_data=monta_arquivo(indx_bag1)
    ind_out=[]
    ind_out.append(str(nome_individuo))
    inst = 0
    inst2 = len(y_data)

    if (geracao == 0 and off == []):
        ind2 = random.randint(1, 100)
        #print("entrei")
    else:
       # print("entreiIIIIIIIIIIII")
        ind2 = random.sample(off, 1)
        ind2=ind2[0]

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

    for j in ind_out:
        if (j != ind_out[-1]):
            individuo_arq.write(j)
            individuo_arq.write(" ")
        else:
            individuo_arq.write(j)
    ind[0] = nome_individuo
    nome_individuo += 1
    if (dispersao==True):
        contador_cruzamento = contador_cruzamento + 1
        if (contador_cruzamento == numero_individuo+1):
            individuo_arq.close()
            contador_cruzamento = 1
            distancia(primeira=False, population=None)
    return ind,

def fitness_f1_n2(individuo):
    global classes
   #print("fitness")
    indx_individuo=abre_arquivo(individuo[0])
    X_data,y_data=monta_arquivo(indx_individuo)

    scaler = MinMaxScaler()
    scaler.fit(X_data)
    transformed_data=scaler.transform(X_data)

    complex=dcol.PPcol(classes=classes)
    complexidades=complex.xy_measures(transformed_data,y_data)
    F1 = np.average(complexidades['F1'])
    N2 = np.average(complexidades['N2'])
    perc=perceptron.Perceptron()
    perc.fit(X_data,y_data)
    score = perc.score(X_valida,y_valida)
    float(score)
    return score, F1, N2,


def fitness_dispercao(individuo):

    global dist, classes
    indx_individuo = abre_arquivo(individuo[0])
    X_data, y_data = monta_arquivo(indx_individuo)
    perc = perceptron.Perceptron()
    perc.fit(X_data, y_data)
    score = perc.score(X_valida, y_valida)
    float(score)
    ind = individuo[0]
    for i in range(len(dist['nome'])):
        if dist['nome'][i][0] == ind:


            dista = dist['dist'][i]
            #print(dist['nome'])
            #print(dist['dist'])
            #print(dist['nome'][i][0], ind)
            #print(dista)
            #exit(0)
            break
    #print(dist['nome'], ind)
    return score, float(dista),

def sequencia():
    global seq
    seq += 1
    return seq

def the_function(population, gen, offspring):
    '''
    responsavel por alterar a geracao, assim como zerar variaveis, alterar populacoes, e copiar arquivos
    :param population: populacao, retorna do DEAP
    :param gen: geracao Retorna do DEAP
    :param offspring: nova populacao
    :return:
    '''
    global geracao, off, X_valida, y_valida, caminho_bags, dispersao
    #print("the_fuction", (population))
    off=[]
    geracao = gen
    #print(population)
    if(geracao==30 and dispersao==True):
        geracao_arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + "-3.indx",
                         'a')
    else:
        geracao_arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indx",
                           'a')
    for i in range(len(population)):
        off.append(population[i][0])
    for j in population:
      #  print (j)
        arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao - 1) + ".indx")
        for i in arq:
            texto = i
           # print(str(j[0]), texto.split(" ")[0])
            if (str(j[0]) == texto.split(" ")[0]):
                #print(i)
                geracao_arq.write(i)
                arq.close()
                break
    geracao_arq.close()
    indx_valida = abre_arquivo(valida=True)
    X_valida, y_valida = monta_arquivo(indx_valida)
    if (dispersao==True):
        distancia(population=population)
    #exit(0)


def populacao(populacao_total):
    '''
    callback para offspring
    :param populacao_total: offspring DEAP
    :return: off (offspring+populacao)
    '''
    global off
    off=[]
    for i in range(len(populacao_total)):
        j=([i][0])
        off.append(j)
    return off
#caminho_originais="/media/marcos/Data/Tese/GA2/Originais/"
caminho_bags="/media/marcos/Data/Tese/GA3/"
caminho_base="/media/marcos/Data/Tese/Bases2/"
off=[]
numero_individuo=500
dispersao=True
contador_cruzamento=1
nome_base="Wine"

nr_generation = 30
proba_crossover = 0.99
proba_mutation = 0.01

fit_value1 = 1.0
fit_value2 = 1.0
#fit_value3 = 1.0

for t in range(1,21):
    classes = []
    off = []
    nome_individuo=101
    repeticao=t
    print("iteracao", t, nome_base)
    geracao=0
    seq=0

    indx_valida=abre_arquivo(valida=True)
    X_valida,y_valida=monta_arquivo(indx_valida,vet_class=True)

    creator.create("FitnessMulti", base.Fitness, weights=(fit_value1,fit_value2))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("map", futures.map)
    toolbox.register("attr_item", sequencia)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, 1)
    population=toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=100)
    distancia(primeira=True)

    toolbox.register("evaluate", fitness_dispercao)
    toolbox.register("mate", cruza)
    toolbox.register("mutate", mutacao)
    toolbox.register("select", tools.selSPEA2 )
    algorithms.eaMuPlusLambda(pop, toolbox, 100, numero_individuo, proba_crossover, proba_mutation, nr_generation, generation_function=the_function, popu=populacao)


