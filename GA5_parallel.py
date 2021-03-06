import Marff
import random, os
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import collections
from sklearn.externals.joblib import Parallel, delayed
import Cpx

def distancia(primeira=False, population=None):

    global pop, nome_individuo, dist,  bags,  min_score, grupo, tipos

    if (primeira==True and geracao == 0):
        print('primeira')
        dist = dict()
        dist['nome'] = pop
        dist['dist'] = list()
       ################
        dist['score']=list()
        ###############
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i,bags,grupo,tipos) for i in range(len(dist['nome'])))
        c, score = zip(*r)
        dist['score']=score
        dist['dist']=Cpx.dispersion(c)
        #print(dist['dist'])
       # exit(0)
        return

    if (primeira == False and population == None):

        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        ############################
        dist['score']=list()
        #############################
        inicio = nome_individuo - numero_individuo
        print("diferente")
        for i in range(inicio, nome_individuo):
            x = []
            x.append(i)
            dist['nome'].append(x)
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(j,bags,grupo,tipos) for j in range(100, numero_individuo + 100))
        c, score = zip(*r)
        dist['dist'] = Cpx.dispersion(c)
       # print(dist['dist'])
        dist['score']=score

        return

    if (population != None):
        print("populacao the function")
        dist = dict()
        dist['nome'] = population
        dist['dist'] = list()
        #######################
        dist['score']=list()
        ######################
        indices=[]
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i,bags,grupo,tipos) for i in indices)
        c, score = zip(*r)
        dist['dist'] = Cpx.dispersion(c)
        dist['score']=score
        return

def parallel_distance(i,bags):

    indx_bag1 = bags['inst'][i]

    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx=(Cpx.complexity_data2(X_bag, y_bag))
    _, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)

    return cpx,score

def parallel_distance2(i,bags,grupo, tipos):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = monta_arquivo(indx_bag1)
    cpx=(Cpx.complexity_data3(X_bag, y_bag,grupo,tipos))
    #print(cpx)
    _, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)
    return cpx,score

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
    global nome_individuo, contador_cruzamento, numero_individuo, bags, dispersao

    individual=False
    indx=bags['nome'].index(str(ind1[0]))
    indx2 = bags['nome'].index(str(ind2[0]))
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
    if len(counter.values())==len(classes) and min(counter.values())>=2:
        return True
    else:return False

def mutacao(ind):
    global off, nome_individuo,  contador_cruzamento, numero_individuo, bags, dispersao
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
        if (contador_cruzamento == numero_individuo + 1):

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
    out=dst+score
    return out,

def fitness_dispercao(ind1):
    global dist, min_score
    for i in range(len(dist['nome'])):
        if (dist['nome'][i][0] == ind1[0]):
           dst = dist['dist'][i]

           ###########################
           score=dist["score"][i]
           print(dist['nome'][i][0],dst,score)
           break
    ###############################

    return dst, score

def fitness_dispercao_linear(ind1):

    global dist
    for i in range(len(dist['nome'])):
       # print(dist['nome'][i][0],ind1[0])
        if (dist['nome'][i][0] == ind1[0]):
            dst1 = dist['dist'][i][0]
            dist2= dist['dist'][i][1]
            ###########################
            score = dist["score"][i]
            break
    ###############################
    print(dst1, dist2, score)

    return dst1, dist2, score,

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
    global geracao, off, dispersao, nr_generation, bags, local, arquivo_de_saida
    print("the_fuction")
    off = []
    geracao = gen
    gg=gen
    base_name=nome_base+str(geracao)
    bags2=bags
    bags=dict()
    bags['nome']=list()
    bags['inst']=list()
    for j in population:
        indx = bags2['nome'].index(str(j[0]))

        bags['nome'].append(bags2['nome'][indx])
        bags['inst'].append(bags2['inst'][indx])

    del bags2
    for i in range(len(population)):
        off.append(population[i][0])

    # for j in off:
    #     print(len((off)))
    #     name = []
    #     indx = bags['nome'].index(str(j))
    #
    #     nm = bags['inst'][indx]
    #     print(len(nm))
    #     name.append(bags['nome'][indx])
    #     name.extend(nm)
    #
    #     Cpx.sprint(off)ave_bag(name, 'bags', local + "/Bags", base_name + arquivo_de_saida+str(gg), repeticao)

    print(dist['score'])
    print(off)
    print(len(off), len(dist['score']))
    exit(0)
    if (gg==nr_generation):
        for j in off:
            name=[]
            indx=bags['nome'].index(str(j))

            nm=bags['inst'][indx]
            name.append(bags['nome'][indx])
            name.extend(nm)
            nm = bags['inst'][indx]

            Cpx.save_bag(name,'bags',local+"/Bags",base_name+arquivo_de_saida,repeticao)

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
nome_base = 'Wine'
#nome_base=sys.argv[1]
#nome_base=sys.argv[1]
#tipos=sys.argv[2]
#tipos=tipos.split(",")
jobs=4
nr_generation = 20
proba_crossover = 0.99
proba_mutation = 0.01

arquivo_de_saida="testegif"

fit_value1 = 1.0
fit_value2 = 1.0
#fit_value3=1.0
local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
min_score=0

#local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
#local = "/home/projeto/Marcos/Bases3"
#caminho_base = "/home/projeto/Marcos/Bases2/"
#cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"

########
#overlapping.F1'	overlapping.F1v'	overlapping.F2'	overlapping.F3'	overlapping.F4'
# neighborhood.N1'	neighborhood.N2'	neighborhood.N3'	neighborhood.N4'	neighborhood.T1'	neighborhood.LSCAvg'
# linearity.L1'	linearity.L2'	linearity.L3'	000000.T2'
# dimensionality.T3'	dimensionality.T4'
# balance.C1'	balance.C2'
# network.Density'	network.ClsCoef'	network.Hubs'
grupo=["overlapping",'neighborhood','','','','']#ATECAO TESTEI PARA DUAS MEDIDAS DE DIFERENTES GRUPOS SOMENTE
tipos=["F3",'N4','','','','']
dispersao = True
for t in range(1, 2):
    classes = []
    off = []
    nome_individuo = 100
    repeticao = t
    print("iteracao", t, nome_base)

    geracao = 0
    seq = -1
    arq_dataset = local_dataset + nome_base + ".arff"
    arq_arff = Marff.abre_arff(arq_dataset)
    X, y, _ = Marff.retorna_instacias(arq_arff)
    _, classes = Marff.retorna_classes_existentes(arq_arff)

    if os.path.isfile(local+"/Bags/"+str(repeticao)+"/"+nome_base+".csv")==False:
        print("entrei")
    ##################Criar bags############################################
        X_train, y_train, X_test, y_test, X_vali, y_vali, dic = Cpx.routine_save_bags(local_dataset, local, nome_base,
                                                                                      repeticao)
    #########################################################################
    else:
        _, validation=Cpx.open_test_vali(local+"/",nome_base,repeticao)
        X_vali,y_vali=Cpx.biuld_x_y(validation,X,y)
    # exit(0)
    bags = Cpx.open_bag(cpx_caminho + str(repeticao) + "/", nome_base)
    creator.create("FitnessMult", base.Fitness, weights=(fit_value1,fit_value2))
    creator.create("Individual", list, fitness=creator.FitnessMult)
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
    toolbox.register("select", tools.selNSGA2)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    pop,log=algorithms.eaMuPlusLambda(pop, toolbox, 100, numero_individuo, proba_crossover, proba_mutation, nr_generation,
                             stats=stats, generation_function=the_function, popu=populacao, verbose=True)

