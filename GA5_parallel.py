import Marff
import random, os
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import collections
from sklearn.externals.joblib import Parallel, delayed

#os.environ['R_HOME'] = '/home/marcos/anaconda3/envs/tese2/lib/R'

import time, Cpx



def distancia(primeira=False, population=None):

    global pop, nome_individuo, dist,  bags,  min_score
    cpx = []
    min_score=0

    if (primeira==True and geracao == 0):
       # print("entrei")
        print('primeira')
        dist = dict()
        dist['nome'] = pop
        dist['dist'] = list()
       ################
       # dist['score']=list()
        ###############

        c = Parallel(n_jobs=-2,verbose=5)(delayed(parallel_distance)(i,bags) for i in range(len(dist['nome'])))

        dist['dist']=Cpx.dispersion(c)

        return

    if (primeira == False and population == None):

        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        ############################
        #dist['score']=list()
        #############################
        inicio = nome_individuo - numero_individuo
        print("diferente")

        for i in range(inicio, nome_individuo):
            x = []
            x.append(i)
            dist['nome'].append(x)

        d = Parallel(n_jobs=-2,verbose=5)(delayed(parallel_distance)(j,bags) for j in range(100, numero_individuo + 100))
        dist['dist'] = Cpx.dispersion(d)

        return

    if (population != None):
        print("populacao the function")
        dist = dict()
        dist['nome'] = population
        dist['dist'] = list()
        #######################
        #dist['score']=list()
        ######################
        indices=[]
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
        c = Parallel(n_jobs=-2,verbose=5)(delayed(parallel_distance)(i,bags) for i in indices)


        dist['dist'] = Cpx.dispersion(c)
       # exit(0)
        return

def parallel_distance(i,bags):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx=(Cpx.complexity_data2(X_bag, y_bag))

    return cpx

def monta_arquivo(indx_bag):
    global X, y
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
    global nome_base, classes, caminho_base
    X_data = []
    y_data = []
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
    global nome_individuo, contador_cruzamento, numero_individuo, bags, dispersao

    individual=False
    indx=bags['nome'].index(str(ind1[0]))
    indx2 = bags['nome'].index(str(ind2[0]))
    #print(indx, indx2)
    indx_bag1 = bags['inst'][indx]
    indx_bag2 = bags['inst'][indx2]
    _, y_data = monta_arquivo(indx_bag1)
    cont=0
    while (individual!=True):

        ind_out1=short_cross(y_data,indx_bag1,indx_bag2)
        individual=verifica_data(ind_out1)
        cont = cont + 1
        if cont==30:
            print("erro de numero de classes")
            exit(0)
    ind1[0] = nome_individuo
    ind2[0] = nome_individuo
    bags['nome'].append(str(nome_individuo))

    bags['inst'].append(ind_out1)
    nome_individuo += 1
    if (dispersao == True):
        contador_cruzamento = contador_cruzamento + 1
        if (contador_cruzamento == numero_individuo + 1):
            contador_cruzamento = 1
            distancia(primeira=False, population=None)
    return creator.Individual(ind1), creator.Individual(ind2)

def short_cross(y_data,indx_bag1,indx_bag2):
    inicio = fim = 0
    ind_out1=[]
    while (y_data[inicio] == y_data[fim]):
        inicio = random.randint(0, len(y_data) - 1)
        fim = random.randint(inicio, len(y_data) - 1)
    for i in range(len(y_data)):
        if (i <= inicio or i >= fim):
            ind_out1.append(indx_bag1[i])
        else:
            ind_out1.append(indx_bag2[i])
    return ind_out1

def verifica_data(ind_out):
    global classes
    _,y=monta_arquivo(ind_out)
    counter = collections.Counter(y)
    #print(counter.values(), min(counter.values()))
    if len(counter.values())==len(classes) and min(counter.values())>=2:
        return True
    else:return False


    #exit(0)

def mutacao(ind):
    global off, nome_individuo,  contador_cruzamento, numero_individuo, bags, dispersao
    #   print("mutacao")
    # print("off", (off))
    # individuo_arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indx",
    #                     'a')
    print("mutacao")
    ind_out = []
    indx = bags['nome'].index(str(ind[0]))
    # for i in range(len(bags['nome'])):
    #     if (bags['nome'][i] == str(ind[0])):
    indx_bag1 = bags['inst'][indx]
    X, y_data = monta_arquivo(indx_bag1)
    # ind_out=[in

    inst = 0
    inst2 = len(y_data)

    if (geracao == 0 and off == []):
        ind2 = random.randint(0, 99)
        # print("entrei")
    else:
        # print("entreiIIIIIIIIIIII")
        ind2 = random.sample(off, 1)
        ind2 = ind2[0]

    indx2 = bags['nome'].index(str(ind2))
    indx_bag2 = bags['inst'][indx2]
    X2, y2_data = monta_arquivo(indx_bag2)

    while y_data[inst] != y2_data[inst2 - 1]:
        inst = random.randint(0, len(y_data) - 1)

    for i in range(len(indx_bag1)):
        if (i == inst):
            ind_out.append(indx_bag2[i])
        else:
            ind_out.append(indx_bag1[i])

    bags['nome'].append(str(nome_individuo))
    bags['inst'].append(ind_out)

    ind[0] = nome_individuo
    nome_individuo += 1
    if (dispersao == True):
        contador_cruzamento = contador_cruzamento + 1
        if (contador_cruzamento == numero_individuo + 1):

            contador_cruzamento = 1
            distancia(primeira=False, population=None)

    return ind,

def fitness_f1_n2(ind1):
    global classes

    for i in range(len(bags['nome'])):
        if (bags['nome'][i] == str(ind1[0])):
            indx_bag1 = bags['inst'][i]

    X_data, y_data = monta_arquivo(indx_bag1)

    dfx = pd.DataFrame(X_data, copy=False)
    dfy = robjects.IntVector(y_data)
    F1 = ecol.overlapping(dfx, dfy, measures='F1')
    N2 = ecol.neighborhood(dfx, dfy, measures='N2')
    T1 = ecol.dimensionality(dfx, dfy, measures='T2')

    return F1[0], N2[0], T1[0],

def fitness_dispercao(ind1):
    global dist, min_score
    for i in range(len(dist['nome'])):
       #
       # (i)
        if (dist['nome'][i][0] == ind1[0]):
           dst = dist['dist'][i]
           ###########################
    #        score=dist["score"][i]
    #        break
    # if score<=min_score:
    #     dst=0.0
    ###############################
    return dst,

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
    global geracao, off, dispersao, nr_generation, bags, local
    print("the_fuction")
    off = []
    geracao = gen
    gg=gen
    #print(gen, nr_generation)
    base_name=nome_base+str(geracao)
    #print(len(bags['inst']))
    bags2=bags
    bags=dict()
    bags['nome']=list()
    bags['inst']=list()
    for j in population:
        indx = bags2['nome'].index(str(j[0]))

        bags['nome'].append(bags2['nome'][indx])
        bags['inst'].append(bags2['inst'][indx])

        #break
    del bags2
    for i in range(len(population)):
        off.append(population[i][0])

    if (gg==nr_generation):
        # for i in bags['inst']:
        #     print(len(i))
        for j in off:
            name=[]
            indx=bags['nome'].index(str(j))

            nm=bags['inst'][indx]
            name.append(bags['nome'][indx])
            #print(len(nm),"\n")
            name.extend(nm)

            Cpx.save_bag(name,'bags',local+"/Bags/",base_name+"sc",repeticao)

    if (dispersao == True and gg!=nr_generation):
        distancia(population=population)

def populacao(populacao_total):
    '''
    callback para offspring
    :param populacao_total: offspring DEAP
    :return: off (offspring+populacao)
    '''
    global off
    off = []
    for i in range(len(populacao_total)):
        j = ([i][0])
        off.append(j)
    return off


off = []
numero_individuo = 100
contador_cruzamento = 1
nome_base = "Haberman"

nr_generation = 20
proba_crossover = 0.99
proba_mutation = 0.01

fit_value1 = 1.0
fit_value2 = 1.0
local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
min_score=0

#local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
#local = "/home/projeto/Marcos/Bases3"
#caminho_base = "/home/projeto/Marcos/Bases2/"
#cpx_caminho="home/projeto/Marcos/Bases3/Bags/"

########
dispersao = True
for t in range(1, 21):
    classes = []
    off = []
    nome_individuo = 100
    repeticao = t
    print("iteracao", t, nome_base)

    geracao = 0
   # print(geracao)
    seq = -1
    ##################Criar bags############################################
    X_train, y_train, X_test, y_test, X_vali, y_vali, dic = Cpx.routine_save_bags(local_dataset, local, nome_base,
                                                                                  repeticao)
    #########################################################################
    arq_dataset = caminho_base + "Dataset/" + nome_base + ".arff"
    arq_arff = Marff.abre_arff(arq_dataset)
    X, y, _ = Marff.retorna_instacias(arq_arff)
    _,classes = Marff.retorna_classes_existentes(arq_arff)
    #print(classes)
    #exit(0)

    bags = Cpx.open_bag(cpx_caminho + str(repeticao) + "/", nome_base)


    creator.create("FitnessMax", base.Fitness, weights=(fit_value1,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_item", sequencia)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, 1)
    population = toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=100)

    if dispersao == True:
        distancia(primeira=True)

    toolbox.register("evaluate", fitness_dispercao)
    toolbox.register("mate", cruza)
    toolbox.register("mutate", mutacao)
    toolbox.register("select", tools.selRoulette)
    algorithms.eaMuPlusLambda(pop, toolbox, 100, numero_individuo, proba_crossover, proba_mutation, nr_generation,
                              generation_function=the_function, popu=populacao)

