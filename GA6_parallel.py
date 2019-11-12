import Marff, numpy as np
import random, os
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import collections, Graficos_ga as graf
from joblib import  Parallel, delayed
import Cpx, sys


def distancia(primeira=False, population=None):
    global pop, nome_individuo, dist, bags, min_score, grupo, tipos

    if (primeira == True and geracao == 0):

        print('primeira')
        dist = dict()
        dist['nome'] = pop
        dist['dist'] = list()
        dist['score'] = list()
        dist['pred'] = list()
        dist['score_v'] = list()

        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i, bags, grupo, tipos) for i in range(len(dist['nome'])))
        c, score, score_val, pred = zip(*r)
        dist['score'] = score

        #dist['dist'] = Cpx.dispersion_norm(c)
        dist['dist'] = Cpx.dispersion_linear(c)
        d = diversidade(pred, y_vali)
        dist['pred']=Cpx.min_max_norm(d)
        dist['score_v']=score_val
        return

    if (primeira == False and population == None):

        print("diferente")
        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        dist['pred'] = list()
        dist['score'] = list()
        dist['score_v'] = list()

        inicio = nome_individuo - nr_individuos

        for i in range(inicio, nome_individuo):
            x = []
            x.append(i)
            dist['nome'].append(x)
        r = Parallel(n_jobs=jobs)(
            delayed(parallel_distance2)(j, bags, grupo, tipos) for j in range(100, nr_individuos + 100))
        c, score, score_val, pred = zip(*r)
        #dist['dist'] = Cpx.dispersion_norm(c)
        dist['dist'] = Cpx.dispersion_linear(c)
        dist['score'] = score
        d = diversidade(pred, y_vali)
        dist['pred'] = Cpx.min_max_norm(d)
        dist['score_v']=score_val
        return

    if (population != None):

        print("populacao the function")
        dist = dict()
        dist['nome'] = population
        dist['dist'] = list()
        dist['pred'] = list()
        dist['score'] = list()
        dist['score_v']=list()

        indices = []
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i, bags, grupo, tipos) for i in indices)
        c, score, score_val, pred = zip(*r)

       # dist['dist'] = Cpx.dispersion_norm(c)
        dist['dist'] = Cpx.dispersion_linear(c)
        dist['score'] = score
        d = diversidade(pred, y_vali)
        dist['pred'] = Cpx.min_max_norm(d)
        dist['score_v']=score_val
       # print(len(score_val))
       # print(len(score))
       # exit(0)
        return


def diversidade(pred, y):

    pred=np.array(pred)
    d =Cpx.diversitys(y, pred)
    return d


def parallel_distance(i, bags):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx = (Cpx.complexity_data2(X_bag, y_bag))
    _, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)

    return cpx, score


def parallel_distance2(i, bags, grupo, tipos):
    """

    :param i: lista de indices do bag a ser testado
    :param bags: lista com todos os bags
    :param grupo: lista com o nome dos grupos de complexidades ex: [overllaping,,,,,]
    :param tipos: lista com o nome da complexidade
    :return: listas de complexidade, score do prorpio bag, score sobre a validacao, e a predicao sobre a validacao
    """

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx = (Cpx.complexity_data3(X_bag, y_bag, grupo, tipos))

    #######################################################################
    ############usando o treino para avaliar o fitness################3####
    _, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_bag, y_bag)
    _, score_val, pred = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)
    #######################################################################
    return cpx, score, score_val, pred


def parallel_distance3(i, bags, grupo, tipos):
    """

    :param i: lista de indices do bag a ser testado
    :param bags: lista com todos os bags
    :param grupo: lista com o nome dos grupos de complexidades ex: [overllaping,,,,,]
    :param tipos: lista com o nome da complexidade
    :return: listas de complexidade, score sobre a validacao, e a predicao sobre a validacao
    """

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx = (Cpx.complexity_data3(X_bag, y_bag, grupo, tipos))

    #######################################################################
    ############usando o treino para avaliar o fitness################3####
    #_, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_bag, y_bag)
    _, score_val, pred = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)
    #######################################################################
    return cpx, score_val, pred


def parallel_score(i, bags):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    _, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)
    return score


def monta_arquivo(indx_bag):
    global X, y
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
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
    global nome_individuo, contador_cruzamento, nr_individuos, bags, dispersao

    individual = False
    indx = bags['nome'].index(str(ind1[0]))
    indx2 = bags['nome'].index(str(ind2[0]))
    indx_bag1 = bags['inst'][indx]
    indx_bag2 = bags['inst'][indx2]
    _, y_data = monta_arquivo(indx_bag1)
    cont = 0

    while (individual != True):

        ind_out1 = short_cross(y_data, indx_bag1, indx_bag2)
        individual = verifica_data(ind_out1)
        cont = cont + 1
        if cont == 30:
            print("erro de numero de classes")
            exit(0)

    ind1[0] = nome_individuo
    ind2[0] = nome_individuo

    bags['nome'].append(str(nome_individuo))
    bags['inst'].append(ind_out1)
    nome_individuo += 1

    if (dispersao == True):
        contador_cruzamento = contador_cruzamento + 1

        if (contador_cruzamento == nr_individuos + 1):
            contador_cruzamento = 1
            distancia(primeira=False, population=None)

    return creator.Individual(ind1), creator.Individual(ind2)

def cruza2(ind1, ind2):
    global nome_individuo, contador_cruzamento, nr_individuos, bags, dispersao

    individual = False
    indx = bags['nome'].index(str(ind1[0]))
    indx2 = bags['nome'].index(str(ind2[0]))
    indx_bag1 = bags['inst'][indx]
    indx_bag2 = bags['inst'][indx2]
    _, y_data = monta_arquivo(indx_bag1)

    cont = 0

    ###############################
    indx_bag1 = [int(i) for i in indx_bag1]
    indx_bag2 = [int(i) for i in indx_bag2]


    ###############################
   # print((indx_bag1))
   # print(indx_bag2)
    while (individual != True):
        ind_out1, _ = tools.cxMessyOnePoint(indx_bag1, indx_bag2)
        individual = verifica_data(ind_out1)
        if cont == 30:
            print("erro de numero de classes")
            exit(0)
    ind_out1 = [str(i) for i in ind_out1]
    #print(ind_out1)
    ind1[0] = nome_individuo
    ind2[0] = nome_individuo
    print(ind_out1)
    bags['nome'].append(str(nome_individuo))
    bags['inst'].append(ind_out1)
    nome_individuo += 1

    if (dispersao == True):
        contador_cruzamento = contador_cruzamento + 1

        if (contador_cruzamento == nr_individuos + 1):
            contador_cruzamento = 1
            distancia(primeira=False, population=None)

    return creator.Individual(ind1), creator.Individual(ind2)

def short_cross(y_data, indx_bag1, indx_bag2):
    inicio = fim = 0
    ind_out1 = []
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
    _, y = monta_arquivo(ind_out)
    counter = collections.Counter(y)
    if len(counter.values()) == len(classes) and min(counter.values()) >= 2:
        return True
    else:
        return False


def mutacao(ind):
    global off, nome_individuo, contador_cruzamento, nr_individuos, bags, dispersao
    print("mutacao")
    ind_out = []
    indx = bags['nome'].index(str(ind[0]))
    indx_bag1 = bags['inst'][indx]
    X, y_data = monta_arquivo(indx_bag1)
    inst = 0
    inst2 = len(y_data)

    if (geracao == 0 and off == []):
        ind2 = random.randint(0, 99)

    else:

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
        if (contador_cruzamento == nr_individuos + 1):
            contador_cruzamento = 1
            distancia(primeira=False, population=None)

    return ind,


def fitness_andre(ind1):
    global dist, min_score
    for i in range(len(dist['nome'])):
        if (dist['nome'][i][0] == ind1[0]):
            dst = dist['dist'][i]
            ###########################
            score = dist["score"][i]
            break
    out = dst + score
    return out,


def fitness_dispercao_diver(ind1):
    global dist, min_score
    for i in range(len(dist['nome'])):
        if (dist['nome'][i][0] == ind1[0]):
            dst = dist['dist'][i]

            ###########################
            score = dist["score"][i]
            disv= dist["pred"][i]
            #print(dist['nome'][i][0], dst, score)
            break
    ###############################

    return dst, disv,


def fitness_dispercao(ind1):
    global dist, min_score
    for i in range(len(dist['nome'])):
        if (dist['nome'][i][0] == ind1[0]):
            dst = dist['dist'][i]

            ###########################
            score = dist["score"][i]
            break
    ###############################

    return dst, score


def fitness_dispercao_linear(ind1):
    global dist
    for i in range(len(dist['nome'])):
        # print(dist['nome'][i][0],ind1[0])
        if (dist['nome'][i][0] == ind1[0]):
            dst1 = dist['dist'][i][0]
            dist2 = dist['dist'][i][1]
            ###########################
            #score = dist["score"][i]
            diver=dist['pred'][i]
            break
    ###############################


    return dst1, dist2, diver,


def sequencia():
    global seq
    seq += 1
    return seq


def melhor_general(population, bags, tipo, accuracia_ant=None, bags_ant=None):
    indices = []
    accuracia_atual = []
    if tipo == 1:
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
            r = Parallel(n_jobs=jobs)(delayed(parallel_score)(i, bags) for i in indices)
        print(population)
        print(r)
        print(dist['score'])
        exit(0)
        return r
    else:
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
            r = Parallel(n_jobs=jobs)(delayed(parallel_score)(i, bags) for i in indices)
        for i in indices:
            accuracia_atual.append(dist['score'][i])
        print(accuracia_ant)
        print(r)
        print(accuracia_atual)
        exit(0)
        return r


def the_function(population, gen, fitness):
    '''
    responsavel por alterar a geracao, assim como zerar variaveis, alterar populacoes, e copiar arquivos
    :param population: populacao, retorna do DEAP
    :param gen: geracao Retorna do DEAP
    :param offspring: nova populacao
    :return:
    '''
    #print(fitness)

    global geracao, off, dispersao, nr_generation, bags, local, arquivo_de_saida, accuracia_ant, s
    print("the_fuction")
    off = []
    s = []
    geracao = gen
    gg = gen
    #if geracao==1:
    graf.grafico_disper(nome_base, ["Dist", "Dist2"], fitness[0], fitness[1], repeticao,  geracao, arquivo_de_saida)
    base_name = nome_base + str(geracao)
    bags_ant = bags
    bags = dict()
    bags['nome'] = list()
    bags['inst'] = list()
    for j in population:
        indx = bags_ant['nome'].index(str(j[0]))

        bags['nome'].append(bags_ant['nome'][indx])
        bags['inst'].append(bags_ant['inst'][indx])

    del bags_ant
    for i in range(len(population)):
        off.append(population[i][0])

    if gg == nr_generation:
        for j in off:
            name = []
            indx = bags['nome'].index(str(j))

            nm = bags['inst'][indx]
            name.append(bags['nome'][indx])
            name.extend(nm)
            nm = bags['inst'][indx]

            Cpx.save_bag(name, 'bags', local + "/Bags", base_name + arquivo_de_saida, repeticao)

    if (dispersao == True and gg != nr_generation):
        distancia(population=population)
    return population


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

nome_base = 'Wine'

local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho = "/media/marcos/Data/Tese/Bases3/Bags/"
min_score = 0

# nome_base=sys.argv[1]
#nome_base=sys.argv[1]
#tipos=sys.argv[2]
#tipos=tipos.split(",")

#local_dataset = "/home/marcosmonteiro/Marcos/Bases3/Dataset/"
#local = "/home/marcosmonteiro/Marcos/Bases3"
#caminho_base = "/home/marcosmonteiro/Marcos/Bases3/"
#cpx_caminho="/home/marcosmonteiro/Marcos/Bases3/Bags/"

# local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
# local = "/home/projeto/Marcos/Bases3"
# caminho_base = "/home/projeto/Marcos/Bases2/"
# cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"

########

grupo = ["overlapping", 'neighborhood', '', '', '', '']
tipos = ["F2", 'N3', '', '', '', '']
dispersao = True
fit_value1 = 1.0
fit_value2 = 1.0
fit_value3 = -1.0
jobs = 4
nr_generation = 20
proba_crossover = 0.99
proba_mutation = 0.01
nr_individuos = 100
p=100
nr_filhos=100
contador_cruzamento = 1
iteracoes=20

arquivo_de_saida = "distdiverlinear"

for t in range(1, iteracoes):

    off = []
    nome_individuo = 100
    repeticao = t
    geracao = 0
    seq = -1
    print("iteracao", t, nome_base)
    ######
    #Abre os bags validacao e teste
    #se nao exisitir cria

    arq_dataset = local_dataset + nome_base + ".arff"
    arq_arff = Marff.abre_arff(arq_dataset)
    X, y, _ = Marff.retorna_instacias(arq_arff)
    _, classes = Marff.retorna_classes_existentes(arq_arff)

    if os.path.isfile(local + "Bags/" + str(repeticao) + "/" + nome_base + ".csv") == False:
        print("entrei")
        X_train, y_train, X_test, y_test, X_vali, y_vali, dic = Cpx.routine_save_bags(local_dataset, local, nome_base,
                                                                                      repeticao)
    else:
        _, validation = Cpx.open_test_vali(local , nome_base, repeticao)
        X_vali, y_vali = Cpx.biuld_x_y(validation, X, y)
    bags = Cpx.open_bag(cpx_caminho + str(repeticao) + "/", nome_base)
    ######
   # ref_points = tools.uniform_reference_points(nobj=2, p=100)
    creator.create("FitnessMult", base.Fitness, weights=(fit_value1, fit_value2, fit_value3))
    creator.create("Individual", list, fitness=creator.FitnessMult)
    toolbox = base.Toolbox()
    toolbox.register("attr_item", sequencia)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, 1)
    population = toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=p)

    if dispersao == True:
        distancia(primeira=True)

    toolbox.register("evaluate", fitness_dispercao_linear)
    toolbox.register("mate", cruza)
    toolbox.register("mutate", mutacao)
    toolbox.register("select", tools.selNSGA2)
    # stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, nr_filhos, nr_individuos, proba_crossover, proba_mutation,
                                         nr_generation,
                                         generation_function=the_function)
    print(log)
   # if geracao>nr_generation or geracao==nr_generation:
       # smd3.selecao(nome_base,local,cpx_caminho,arquivo_de_saida,arquivo_de_saida, repeticao)
import Marff, numpy as np
import random, os
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import collections, Graficos_ga as graf
from joblib import  Parallel, delayed
import Cpx, sys


def distancia(primeira=False, population=None):
    global pop, nome_individuo, dist, bags, min_score, grupo, tipos

    if (primeira == True and geracao == 0):

        print('primeira')
        dist = dict()
        dist['nome'] = pop
        dist['dist'] = list()
        dist['score'] = list()
        dist['pred'] = list()
        dist['score_v'] = list()

        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i, bags, grupo, tipos) for i in range(len(dist['nome'])))
        c, score, score_val, pred = zip(*r)
        dist['score'] = score

        #dist['dist'] = Cpx.dispersion_norm(c)
        dist['dist'] = Cpx.dispersion_linear(c)
        d = diversidade(pred, y_vali)
        dist['pred']=Cpx.min_max_norm(d)
        dist['score_v']=score_val
        return

    if (primeira == False and population == None):

        print("diferente")
        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        dist['pred'] = list()
        dist['score'] = list()
        dist['score_v'] = list()

        inicio = nome_individuo - nr_individuos

        for i in range(inicio, nome_individuo):
            x = []
            x.append(i)
            dist['nome'].append(x)
        r = Parallel(n_jobs=jobs)(
            delayed(parallel_distance2)(j, bags, grupo, tipos) for j in range(100, nr_individuos + 100))
        c, score, score_val, pred = zip(*r)
        #dist['dist'] = Cpx.dispersion_norm(c)
        dist['dist'] = Cpx.dispersion_linear(c)
        dist['score'] = score
        d = diversidade(pred, y_vali)
        dist['pred'] = Cpx.min_max_norm(d)
        dist['score_v']=score_val
        return

    if (population != None):

        print("populacao the function")
        dist = dict()
        dist['nome'] = population
        dist['dist'] = list()
        dist['pred'] = list()
        dist['score'] = list()
        dist['score_v']=list()

        indices = []
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i, bags, grupo, tipos) for i in indices)
        c, score, score_val, pred = zip(*r)

       # dist['dist'] = Cpx.dispersion_norm(c)
        dist['dist'] = Cpx.dispersion_linear(c)
        dist['score'] = score
        d = diversidade(pred, y_vali)
        dist['pred'] = Cpx.min_max_norm(d)
        dist['score_v']=score_val
       # print(len(score_val))
       # print(len(score))
       # exit(0)
        return


def diversidade(pred, y):

    pred=np.array(pred)
    d =Cpx.diversitys(y, pred)
    return d


def parallel_distance(i, bags):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx = (Cpx.complexity_data2(X_bag, y_bag))
    _, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)

    return cpx, score


def parallel_distance2(i, bags, grupo, tipos):
    """

    :param i: lista de indices do bag a ser testado
    :param bags: lista com todos os bags
    :param grupo: lista com o nome dos grupos de complexidades ex: [overllaping,,,,,]
    :param tipos: lista com o nome da complexidade
    :return: listas de complexidade, score do prorpio bag, score sobre a validacao, e a predicao sobre a validacao
    """

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx = (Cpx.complexity_data3(X_bag, y_bag, grupo, tipos))

    #######################################################################
    ############usando o treino para avaliar o fitness################3####

    _, score,score_val, pred = Cpx.biuld_classifier(X_bag, y_bag, X_bag, y_bag,X_vali,y_vali)
    #######################################################################
    return cpx, score, score_val, pred


def parallel_distance3(i, bags, grupo, tipos):
    """

    :param i: lista de indices do bag a ser testado
    :param bags: lista com todos os bags
    :param grupo: lista com o nome dos grupos de complexidades ex: [overllaping,,,,,]
    :param tipos: lista com o nome da complexidade
    :return: listas de complexidade, score sobre a validacao, e a predicao sobre a validacao
    """

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx = (Cpx.complexity_data3(X_bag, y_bag, grupo, tipos))

    #######################################################################
    ############usando o treino para avaliar o fitness################3####
    #_, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_bag, y_bag)
    _, score_val, pred = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)
    #######################################################################
    return cpx, score_val, pred


def parallel_score(i, bags):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    _, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)
    return score


def monta_arquivo(indx_bag):
    global X, y
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
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
    global nome_individuo, contador_cruzamento, nr_individuos, bags, dispersao

    individual = False
    indx = bags['nome'].index(str(ind1[0]))
    indx2 = bags['nome'].index(str(ind2[0]))
    indx_bag1 = bags['inst'][indx]
    indx_bag2 = bags['inst'][indx2]
    _, y_data = monta_arquivo(indx_bag1)
    cont = 0

    while (individual != True):

        ind_out1 = short_cross(y_data, indx_bag1, indx_bag2)
        individual = verifica_data(ind_out1)
        cont = cont + 1
        if cont == 30:
            print("erro de numero de classes")
            exit(0)

    ind1[0] = nome_individuo
    ind2[0] = nome_individuo

    bags['nome'].append(str(nome_individuo))
    bags['inst'].append(ind_out1)
    nome_individuo += 1

    if (dispersao == True):
        contador_cruzamento = contador_cruzamento + 1

        if (contador_cruzamento == nr_individuos + 1):
            contador_cruzamento = 1
            distancia(primeira=False, population=None)

    return creator.Individual(ind1), creator.Individual(ind2)

def cruza2(ind1, ind2):
    global nome_individuo, contador_cruzamento, nr_individuos, bags, dispersao

    individual = False
    indx = bags['nome'].index(str(ind1[0]))
    indx2 = bags['nome'].index(str(ind2[0]))
    indx_bag1 = bags['inst'][indx]
    indx_bag2 = bags['inst'][indx2]
    _, y_data = monta_arquivo(indx_bag1)

    cont = 0

    ###############################
    indx_bag1 = [int(i) for i in indx_bag1]
    indx_bag2 = [int(i) for i in indx_bag2]


    ###############################
   # print((indx_bag1))
   # print(indx_bag2)
    while (individual != True):
        ind_out1, _ = tools.cxMessyOnePoint(indx_bag1, indx_bag2)
        individual = verifica_data(ind_out1)
        if cont == 30:
            print("erro de numero de classes")
            exit(0)
    ind_out1 = [str(i) for i in ind_out1]
    #print(ind_out1)
    ind1[0] = nome_individuo
    ind2[0] = nome_individuo
    print(ind_out1)
    bags['nome'].append(str(nome_individuo))
    bags['inst'].append(ind_out1)
    nome_individuo += 1

    if (dispersao == True):
        contador_cruzamento = contador_cruzamento + 1

        if (contador_cruzamento == nr_individuos + 1):
            contador_cruzamento = 1
            distancia(primeira=False, population=None)

    return creator.Individual(ind1), creator.Individual(ind2)

def short_cross(y_data, indx_bag1, indx_bag2):
    inicio = fim = 0
    ind_out1 = []
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
    _, y = monta_arquivo(ind_out)
    counter = collections.Counter(y)
    if len(counter.values()) == len(classes) and min(counter.values()) >= 2:
        return True
    else:
        return False


def mutacao(ind):
    global off, nome_individuo, contador_cruzamento, nr_individuos, bags, dispersao
    print("mutacao")
    ind_out = []
    indx = bags['nome'].index(str(ind[0]))
    indx_bag1 = bags['inst'][indx]
    X, y_data = monta_arquivo(indx_bag1)
    inst = 0
    inst2 = len(y_data)

    if (geracao == 0 and off == []):
        ind2 = random.randint(0, 99)

    else:

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
        if (contador_cruzamento == nr_individuos + 1):
            contador_cruzamento = 1
            distancia(primeira=False, population=None)

    return ind,


def fitness_andre(ind1):
    global dist, min_score
    for i in range(len(dist['nome'])):
        if (dist['nome'][i][0] == ind1[0]):
            dst = dist['dist'][i]
            ###########################
            score = dist["score"][i]
            break
    out = dst + score
    return out,


def fitness_dispercao_diver(ind1):
    global dist, min_score
    for i in range(len(dist['nome'])):
        if (dist['nome'][i][0] == ind1[0]):
            dst = dist['dist'][i]

            ###########################
            score = dist["score"][i]
            disv= dist["pred"][i]
            #print(dist['nome'][i][0], dst, score)
            break
    ###############################

    return dst, disv,


def fitness_dispercao(ind1):
    global dist, min_score
    for i in range(len(dist['nome'])):
        if (dist['nome'][i][0] == ind1[0]):
            dst = dist['dist'][i]

            ###########################
            score = dist["score"][i]
            break
    ###############################

    return dst, score


def fitness_dispercao_linear(ind1):
    global dist
    for i in range(len(dist['nome'])):
        # print(dist['nome'][i][0],ind1[0])
        if (dist['nome'][i][0] == ind1[0]):
            dst1 = dist['dist'][i][0]
            dist2 = dist['dist'][i][1]
            ###########################
            #score = dist["score"][i]
            diver=dist['pred'][i]
            break
    ###############################


    return dst1, dist2, diver,


def sequencia():
    global seq
    seq += 1
    return seq


def melhor_general(population, bags, tipo, accuracia_ant=None, bags_ant=None):
    indices = []
    accuracia_atual = []
    if tipo == 1:
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
            r = Parallel(n_jobs=jobs)(delayed(parallel_score)(i, bags) for i in indices)
        print(population)
        print(r)
        print(dist['score'])
        exit(0)
        return r
    else:
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
            r = Parallel(n_jobs=jobs)(delayed(parallel_score)(i, bags) for i in indices)
        for i in indices:
            accuracia_atual.append(dist['score'][i])
        print(accuracia_ant)
        print(r)
        print(accuracia_atual)
        exit(0)
        return r


def the_function(population, gen, fitness):
    '''
    responsavel por alterar a geracao, assim como zerar variaveis, alterar populacoes, e copiar arquivos
    :param population: populacao, retorna do DEAP
    :param gen: geracao Retorna do DEAP
    :param offspring: nova populacao
    :return:
    '''
    #print(fitness)

    global geracao, off, dispersao, nr_generation, bags, local, arquivo_de_saida, accuracia_ant, s
    print("the_fuction")
    off = []
    s = []
    geracao = gen
    gg = gen
    #if geracao==1:
    graf.grafico_disper(nome_base, ["Dist", "Dist2"], fitness[0], fitness[1], repeticao,  geracao, arquivo_de_saida)
    base_name = nome_base + str(geracao)
    bags_ant = bags
    bags = dict()
    bags['nome'] = list()
    bags['inst'] = list()
    for j in population:
        indx = bags_ant['nome'].index(str(j[0]))

        bags['nome'].append(bags_ant['nome'][indx])
        bags['inst'].append(bags_ant['inst'][indx])

    del bags_ant
    for i in range(len(population)):
        off.append(population[i][0])

    if gg == nr_generation:
        for j in off:
            name = []
            indx = bags['nome'].index(str(j))

            nm = bags['inst'][indx]
            name.append(bags['nome'][indx])
            name.extend(nm)
            nm = bags['inst'][indx]

            Cpx.save_bag(name, 'bags', local + "/Bags", base_name + arquivo_de_saida, repeticao)

    if (dispersao == True and gg != nr_generation):
        distancia(population=population)
    return population


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

nome_base = 'Wine'

local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho = "/media/marcos/Data/Tese/Bases3/Bags/"
min_score = 0

# nome_base=sys.argv[1]
#nome_base=sys.argv[1]
#tipos=sys.argv[2]
#tipos=tipos.split(",")

#local_dataset = "/home/marcosmonteiro/Marcos/Bases3/Dataset/"
#local = "/home/marcosmonteiro/Marcos/Bases3"
#caminho_base = "/home/marcosmonteiro/Marcos/Bases3/"
#cpx_caminho="/home/marcosmonteiro/Marcos/Bases3/Bags/"

# local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
# local = "/home/projeto/Marcos/Bases3"
# caminho_base = "/home/projeto/Marcos/Bases2/"
# cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"

########

grupo = ["overlapping", 'neighborhood', '', '', '', '']
tipos = ["F2", 'N3', '', '', '', '']
dispersao = True
fit_value1 = 1.0
fit_value2 = 1.0
fit_value3 = -1.0
jobs = 4
nr_generation = 20
proba_crossover = 0.99
proba_mutation = 0.01
nr_individuos = 100
p=100
nr_filhos=100
contador_cruzamento = 1
iteracoes=20

arquivo_de_saida = "distdiverlinear"

for t in range(1, iteracoes):

    off = []
    nome_individuo = 100
    repeticao = t
    geracao = 0
    seq = -1
    print("iteracao", t, nome_base)
    ######
    #Abre os bags validacao e teste
    #se nao exisitir cria

    arq_dataset = local_dataset + nome_base + ".arff"
    arq_arff = Marff.abre_arff(arq_dataset)
    X, y, _ = Marff.retorna_instacias(arq_arff)
    _, classes = Marff.retorna_classes_existentes(arq_arff)

    if os.path.isfile(local + "Bags/" + str(repeticao) + "/" + nome_base + ".csv") == False:
        print("entrei")
        X_train, y_train, X_test, y_test, X_vali, y_vali, dic = Cpx.routine_save_bags(local_dataset, local, nome_base,
                                                                                      repeticao)
    else:
        _, validation = Cpx.open_test_vali(local , nome_base, repeticao)
        X_vali, y_vali = Cpx.biuld_x_y(validation, X, y)
    bags = Cpx.open_bag(cpx_caminho + str(repeticao) + "/", nome_base)
    ######
   # ref_points = tools.uniform_reference_points(nobj=2, p=100)
    creator.create("FitnessMult", base.Fitness, weights=(fit_value1, fit_value2, fit_value3))
    creator.create("Individual", list, fitness=creator.FitnessMult)
    toolbox = base.Toolbox()
    toolbox.register("attr_item", sequencia)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, 1)
    population = toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=p)

    if dispersao == True:
        distancia(primeira=True)

    toolbox.register("evaluate", fitness_dispercao_linear)
    toolbox.register("mate", cruza)
    toolbox.register("mutate", mutacao)
    toolbox.register("select", tools.selNSGA2)
    # stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, nr_filhos, nr_individuos, proba_crossover, proba_mutation,
                                         nr_generation,
                                         generation_function=the_function)
    print(log)
   # if geracao>nr_generation or geracao==nr_generation:
       # smd3.selecao(nome_base,local,cpx_caminho,arquivo_de_saida,arquivo_de_saida, repeticao)
