import Marff
import random, os
import novo_perceptron as perce
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.preprocessing import MinMaxScaler
import complexity_pcol as dcol
import numpy as np
from math import sqrt

os.environ['R_HOME'] = '/home/marcos/anaconda3/envs/tese2/lib/R'
import pandas as pd
from rpy2.robjects import pandas2ri

pandas2ri.activate()
import rpy2.robjects.packages as rpackages

ecol = rpackages.importr('ECoL')
import rpy2.robjects as robjects
import time, Cpx

inicio = time.time()


def altera_arquivo_geração():
    '''
    da nome aos bags nesse caso 1 a 100
    :return: altera os arquivos
    '''
    global repeticao, nome_base, geracao, caminho_bags, caminho_originais
    arq = open(caminho_bags + str(repeticao) + "/" + nome_base + geracao + "-9.indx")
    arqtemp = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indxTemp", 'w')
    cont = 1
    for i in arq:
        texto = i
        q = texto.split(" ")
        q = q[1:]
        q.insert(0, str(cont))
        # print(q)
        for j in q:
            # print(j)
            if (j != q[-1]):
                arqtemp.write(j)
                arqtemp.write(" ")
            else:
                arqtemp.write(j)
                # arqtemp.write('\n')
        cont += 1
    arq.close()
    arqtemp.close()
    os.system(
        "cp -r " + caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indxTemp " + caminho_bags + str(
            repeticao) + "/" + nome_base + str(geracao) + ".indx")
    os.system("rm " + caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indxTemp")


def inter():
    global X, bags
    inter = dict()
    inter['nome'] = bags['nome']
    inter['n_int'] = np.zeros((len(X),), dtype=np.int)


def distancia(primeira=False, population=None):
    global classes, off, pop, nome_individuo, dist, geracao, dispersao, bags, dic
    cpx = []
    if (geracao == 30 and dispersao == True):
        None
    if (primeira and geracao == 0):
        print('primeira')
        dist = dict()
        dist['nome'] = pop
        dist['dist'] = list()
        for i in dist['nome']:
            x = i[0]
            # print(bags['inst'][x])
            indx_bag1 = bags['inst'][x]
            X_bag, y_bag = monta_arquivo(indx_bag1)
            # print(type(X_bag[0]))
            w = Cpx.biuld_dic(X_bag, y_bag, dic)
            # print(w)
            # exit(0)
            Cpx.generate_csv(w)
            cpx.append(Cpx.complexity_data())
        dist['dist'] = Cpx.dispersion(cpx)
       # exit(0)
        return dist
        # print('tamanho das dist inicial', len(dist['dist']))

    if (population != None):
        print("populacao")
        dist = dict()
        dist['nome'] = population
        dist['dist'] = list()
        for i in population:
            for j in range(len(bags['nome'])):
                if i[0] == bags['inst'][j]:
                    indx_bag = bags['inst'][j]
                    X, y = monta_arquivo(indx_bag)
                    w = Cpx.biuld_dic(X, y, dic)
                    Cpx.generate_csv(w)
                    cpx.append(Cpx.complexity_data())
        dist['dist'] = Cpx.dispersion(cpx)
        return dist

    if (primeira == False and population == None):

        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        inicio = nome_individuo - numero_individuo
        print("diferente")
        dist['nome'] = range(inicio, nome_individuo)
        for j in range(100, numero_individuo + 100):
            # print(j.split(" ")[0])
            # print(j)
            indx_bag = bags['inst'][j]
            # print(indx_bag[0])

            X, y = monta_arquivo(indx_bag)
            w = Cpx.biuld_dic(X, y, dic)
            Cpx.generate_csv(w)
            cpx.append(Cpx.complexity_data())
        dist['dist'] = Cpx.dispersion(cpx)
        return dist


def abre_arquivo(individuo=None, valida=False):
    '''
    :param individuo: nome do individuo
    :param valida: para aproveitar a funcao ela abre a validacao
    :return: linha com indices do arquivo, caso do individuo exclui a primeira coluna, validacao nao
    '''
    global nome_base, repeticao, geracao, caminho_bags, caminho_base
    if individuo:
        arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indx")
        # print(arq)
        for i in arq:
            texto = i
            # print('abre arquivo indi',str(individuo))
            if (str(individuo) == texto.split(" ")[0]):
                indx_bag = texto.split(" ")
                arq.close()
                indx_bag = indx_bag[1:]
                break

    elif valida:
        arq = open(caminho_base + "Validacao/" + str(repeticao) + "/" + nome_base + ".idx")
        texto = arq.readline()
        indx_bag = texto.split(" ")
        arq.close()

    # print(indx_bag)
    return indx_bag


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
    global nome_individuo, repeticao, nome_base, geracao, caminho_bags, dispersao, contador_cruzamento, numero_individuo, bags

    inicio = fim = 0
    ind_out1 = []

    for i in range(len(bags['nome'])):
        if (bags['nome'][i] == str(ind1[0])):
            indx_bag1 = bags['inst'][i]
    for i in range(len(bags['nome'])):
        if (bags['nome'][i] == str(ind1[0])):
            indx_bag2 = bags['inst'][i]
    X, y_data = monta_arquivo(indx_bag2)
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
    bags['nome'].append(str(nome_individuo))

    bags['inst'].append(ind_out1)
    nome_individuo += 1
    if (dispersao == True):
        contador_cruzamento = contador_cruzamento + 1
        if (contador_cruzamento == numero_individuo + 1):
            contador_cruzamento = 1
            distancia(primeira=False, population=None)
    # print(len(bags['nome']), len(bags['inst']))
    return creator.Individual(ind1), creator.Individual(ind2)


def mutacao(ind):
    global off, nome_individuo, dispersao, contador_cruzamento, numero_individuo, bags
    #   print("mutacao")
    # print("off", (off))
    # individuo_arq = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indx",
    #                     'a')
    ind_out = []
    for i in range(len(bags['nome'])):
        if (bags['nome'][i] == str(ind[0])):
            indx_bag1 = bags['inst'][i]
    X, y_data = monta_arquivo(indx_bag1)
    # ind_out=[]

    inst = 0
    inst2 = len(y_data)

    if (geracao == 0 and off == []):
        ind2 = random.randint(1, 100)
        # print("entrei")
    else:
        # print("entreiIIIIIIIIIIII")
        ind2 = random.sample(off, 1)
        ind2 = ind2[0]

        # print('mutacaooooo e contador', individuo, ind2, contador_cruzamento)
    for i in range(len(bags['nome'])):
        if (bags['nome'][i] == str(ind2)):
            indx_bag2 = bags['inst'][i]
    # indx_bag2 = abre_arquivo(ind2)
    X2, y2_data = monta_arquivo(indx_bag2)

    while y_data[inst] != y2_data[inst2 - 1]:
        inst = random.randint(0, len(y_data) - 1)
        inst2 = random.randint(0, len(y2_data) - 1)
    for i in range(len(indx_bag1)):
        if (i == inst):
            ind_out.append(indx_bag2[i])
        else:
            ind_out.append(indx_bag1[i])
            # print(ind_out)
    bags['nome'].append(str(nome_individuo))
    bags['inst'].append(ind_out)
    # print(bags['inst'][-1])
    # print(len(bags['nome']), len(bags['inst']))
    # exit(0)

    ind[0] = nome_individuo
    nome_individuo += 1
    if (dispersao == True):
        contador_cruzamento = contador_cruzamento + 1
        if (contador_cruzamento == numero_individuo + 1):

            contador_cruzamento = 1
            distancia(primeira=False, population=None)
    # print(len(bags['nome']),len(bags['inst']))
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
    global dist, classes
    for i in range(len(bags['nome'])):
        if (bags['nome'][i] == str(ind1[0])):
            indx_bag1 = bags['inst'][i]
    X_data, y_data = monta_arquivo(indx_bag1)
    perc, score, predict = Cpx.biuld_classifier(X_data, y_data, X_vali, y_vali)
    return score, float(dist),


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
    global geracao, off, X_valida, y_valida, caminho_bags, dispersao, nr_generation, bags, local
    print("the_fuction", (population))
    off = []
    geracao = gen
    base_name=nome_base+geracao


    for i in range(len(population)):
        off.append(population[i][0])
    if (geracao==nr_generation):
        for j in population:
            for i in range(len(bags['nome'])):
                if (j[0] == bags['nome'][i]):
                    nm=[]
                    nm.append(bags['nome'][i])
                    nm.append(bags['inst'][i])

                    Cpx.save_bag(nm,'bags',local+"/Bags/",base_name,repeticao)

    # indx_valida = abre_arquivo(valida=True)
    # X_valida, y_valida = monta_arquivo(indx_valida)
    # bags = dict()
    # bags['nome'] = list()
    # bags['inst'] = list()
    #
    # arq_bags = open(caminho_bags + str(repeticao) + "/" + nome_base + str(geracao) + ".indx")
    # for i in arq_bags:
    #     text = i
    #     bags['nome'].append(text.split(" ")[0])
    #     temp = text.split(" ")
    #     bags['inst'].append(temp[1:-1])
    if (dispersao == True):
        distancia(population=population)
        # exit(0)


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


caminho_originais = "/media/marcos/Data/Tese/Bases2/Originais/"
caminho_bags = "/media/marcos/Data/Tese/Bases3/bags/"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
# off=[]
# numero_individuo=100
# dispersao=False
# contador_cruzamento=1
# nome_base="Wine"
#
# nr_generation = 20
# proba_crossover = .99
# proba_mutation = 0.01
#
# fit_value1 = 1.0
# fit_value2 = -1.0
# fit_value3 = 1.0
# for i in range(1,2):
# caminho_originais = "/home/projeto/Marcos/Originais/"
# caminho_bags = "/home/projeto/Marcos/GA9/"
# caminho_base = "/home/projeto/Marcos/Bases2/"

off = []
numero_individuo = 100
dispersao = True
contador_cruzamento = 1
nome_base = "Banana"
comple = "F1,N2,T2"

nr_generation = 10
proba_crossover = 0.99
proba_mutation = 0.01

fit_value1 = -1.0
fit_value2 = 1.0
# fit_value3 = -1.0
local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3"




########

for t in range(1, 21):
    classes = []
    off = []
    nome_individuo = 101
    repeticao = t
    print("iteracao", t, nome_base)

    geracao = 0
    print(geracao)
    seq = 0

    X_train, y_train, X_test, y_test, X_vali, y_vali, dic = Cpx.routine_save_bags(local_dataset, local, nome_base,
                                                                                  repeticao)
    bags = Cpx.open_bag("/media/marcos/Data/Tese/Bases3/Bags/" + str(repeticao) + "/", nome_base)
    arq_dataset = caminho_base + "Dataset/" + nome_base + ".arff"
    arq_arff = Marff.abre_arff(arq_dataset)
    X, y, _ = Marff.retorna_instacias(arq_arff)
    bags = Cpx.open_bag("/media/marcos/Data/Tese/Bases3/Bags/" + str(repeticao) + "/", nome_base)

    creator.create("FitnessMulti", base.Fitness, weights=(fit_value1, fit_value2,))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_item", sequencia)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, 1)
    population = toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=100)

    if dispersao == True:
        distancia(primeira=True)

    toolbox.register("evaluate", dispersao)
    toolbox.register("mate", cruza)
    toolbox.register("mutate", mutacao)
    toolbox.register("select", tools.selNSGA2)
    algorithms.eaMuPlusLambda(pop, toolbox, 100, numero_individuo, proba_crossover, proba_mutation, nr_generation,
                              generation_function=the_function, popu=populacao)
fim = time.time()
print(fim - inicio)
arq_descricoes = open("/home/projeto/Marcos/Descricoes.txt", "a")
arq_descricoes.write(
    "nome da base,numero de geracoes, tipo de fit1,2,3, fitness_dispersao, caminho bag, quais complexidades\n")
arq_descricoes.write(
    " {},{},{},{},{},{},{}\n".format(nome_base, nr_generation, fit_value1, fit_value2, caminho_bags,
                                     comple))
arq_descricoes.close()
