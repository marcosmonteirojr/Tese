import Marff, numpy as np
import random, os
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import collections, Graficos_ga as graf
from joblib import  Parallel, delayed
import Cpx, sys, csv, time


def distancia(primeira=False, population=None):
    global pop, nome_individuo, dist, bags, min_score, grupo, tipos, c
    dist = dict()
    dist['nome'] = list()
    dist['dist'] = list()
    dist['diver'] = list()
    dist['score'] = list()
    dist['score_g'] = list()
    if (primeira == True and generation == 0):
        dist['nome'] = pop
        print('primeira')
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i, bags, grupo, tipos) for i in range(len(dist['nome'])))
        c, score,  pred, pool = zip(*r)


    elif (primeira == False and population == None):

        print("diferente")
        inicio = nome_individuo - nr_individual
        for i in range(inicio, nome_individuo):
            x = []
            x.append(i)
            dist['nome'].append(x)
        r = Parallel(n_jobs=jobs)(
            delayed(parallel_distance2)(j, bags, grupo, tipos) for j in range(100, nr_individual + 100))
        c, score, pred, pool = zip(*r)

    elif (population != None):

        print("populacao the function")

        dist['nome'] = population
        indices = []
        for i in population:
            indices.append(bags['nome'].index(str(i[0])))
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance2)(i, bags, grupo, tipos) for i in indices)
        c, score, pred, pool = zip(*r)

    #dist['dist'] = Cpx.dispersion_norm(c)


    dist['dist'] = Cpx.dispersion_linear(c)
    dist['score'] = score
    d = diversidade(pred, y_vali)
    dist['diver'] = Cpx.min_max_norm(d)
    dist['score_g']=Cpx.voting_classifier(pool,X_vali,y_vali)

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
   # _, score, _ = Cpx.biuld_classifier(X_bag, y_bag, X_bag, y_bag)
    perc, score, pred = Cpx.biuld_classifier(X_bag, y_bag, X_bag, y_bag,X_vali,y_vali)
    #######################################################################
    return cpx,  score,  pred, perc

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
    global nome_individuo, cont_crossover, nr_individual, bags, dispersao

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
        cont_crossover = cont_crossover + 1

        if (cont_crossover == nr_individual + 1):
            cont_crossover = 1
            distancia(primeira=False, population=None)

    return creator.Individual(ind1), creator.Individual(ind2)

def cruza2(ind1, ind2):
    global nome_individuo, cont_crossover, nr_individual, bags, dispersao

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
        cont_crossover = cont_crossover + 1

        if (cont_crossover == nr_individual + 1):
            cont_crossover = 1
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
    global off, nome_individuo, cont_crossover, nr_individual, bags, dispersao
    print("mutacao")
    ind_out = []
    indx = bags['nome'].index(str(ind[0]))
    indx_bag1 = bags['inst'][indx]
    X, y_data = monta_arquivo(indx_bag1)
    inst = 0
    inst2 = len(y_data)

    if (generation == 0 and off == []):
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
        cont_crossover = cont_crossover + 1
        if (cont_crossover == nr_individual + 1):
            cont_crossover = 1
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
            disv= dist["diver"][i]
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
            diver=dist['diver'][i]
            break
    ###############################


    return dst1, dist2, diver,

def sequencia():
    global seq
    seq += 1
    return seq

def the_function(population, gen, fitness):

    '''
    responsavel por alterar a generation, assim como zerar variaveis, alterar populacoes, e copiar arquivos
    :param population: populacao, retorna do DEAP
    :param gen: generation Retorna do DEAP
    :param offspring: nova populacao
    :return:
    '''

    global generation, off, dispersao, nr_generation, bags, local, arquivo_de_saida, accuracia_ant, \
        s, c, dist_temp, gen_temp, pop_temp, bags_temp, parada, salva_info

    generation = gen
    ###############################################333
    if salva_info:
        if repeticao == 1:
           salva_informacoes_geracoes(generation,fitness,c)
    ###################################################3
    print("the_fuction")

    off = []
    base_name = nome_base + str(generation)
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

    if parada=="maxdistance":
        max_distancia(fitness, generation=generation, population=off, bags=bags)
    elif parada =="maxacc":
        max_acc(dist['score_g'], generation=generation, population=off, bags=bags)

    if generation == nr_generation:

        if parada=="maxdistance":
            salva_bags(pop_temp, bags_temp, gen_temp, base_name, tipo=1)
        elif parada=="maxacc":
            salva_bags(pop_temp, bags_temp, gen_temp, base_name, tipo=2)
        else:
            salva_bags(off,bags,base_name=base_name,tipo=0)

    if (dispersao == True and generation != nr_generation):
        distancia(population=population)
    return population

def salva_informacoes_geracoes(generation,fitness, complexidade):
    '''

    :param generation:
    :param fitness:
    :param complexidade:
    :return:
    sava em arquivo todos os dados da geração, o ultimo comando salva um grafíco dos 2 primeiros fitness (1d) com o nome do arquivo de saida
    '''

    global nome_base

    temp = np.array(complexidade)
    temp=temp.T
    k = []
    with open(nome_base +str(generation)+ arquivo_de_saida+'.csv', 'a', newline='') as csvfile:
          spamwriter = csv.writer(csvfile, delimiter=';')
          for i in range(len(complexidade)):
              k=complexidade[i]
              k.append(fitness[0][i])
              k.append(fitness[1][i])
              if fitness[2]:
                k.append(fitness[2][i])
              spamwriter.writerow(k)

          m1 = np.std(temp[0])
          m2 = np.std(temp[1])
          m3 = np.std(fitness[0])
          m4 = np.std(fitness[1])

          m6 = np.mean(temp[0])
          m7 = np.mean(temp[1])
          m8 = np.mean(fitness[0])
          m9 = np.mean(fitness[1])
          if fitness[2]:
            m10 = np.mean(fitness[2])
            m5 = np.std(fitness[2])
            tem=[m1,m2, m3, m4, m5, m6, m7,m8, m9, m10]
            tem2.append(tem)
            spamwriter.writerow(tem)
            spamwriter.writerow('\n')
            del m5, m10
          else:
              tem = [m1, m2, m3, m4, m6, m7, m8, m9]
              tem2.append(tem)
              spamwriter.writerow(tem)
              spamwriter.writerow('\n')
          if generation == nr_generation:
              for i in tem2:
                  spamwriter.writerow(i)

    graf.grafico_disper(nome_base, ["Disp complexity overlapping", "Disp complexity neighborhood", "Disp diversity"], fitness[0], fitness[1], valor3=fitness[2], legend="Lithuanian", i= repeticao, gr=generation,pasta= arquivo_de_saida)
    del tem, k, m1, m2, temp,m3, m4, m6, m7,m8, m9

def salva_bags(pop_temp, bags_temp, gen_temp=None, base_name=None, tipo=0):
    '''

    :param pop_temp: população a ser gravada, geralmente o off
    :param bags_temp: bags a serem gravados, geralmente os bags da the function ou bags da max (bags_temp)
    :param gen_temp: geração atual, ou a geração escolhida (melhor geração) isso soma no nome do arquivo final
    :param base_name: nesse caso o número geração junto ao nome da base (isso soma no arquivo de saida (nome do arquivo final))
    :param tipo: primeiro tipo (0) a população final tradicional, tipo (1) população da distancia média, tipo (2) população da acurácia global
    :return:
    '''
    global nome_base, arquivo_de_saida, repeticao
    if tipo==0:

        for j in pop_temp:
            name = []
            indx = bags['nome'].index(str(j))
            nm = bags['inst'][indx]
            name.append(bags['nome'][indx])
            name.extend(nm)
            Cpx.save_bag(name, 'bags', local + "/Bags", base_name + arquivo_de_saida, repeticao)

    elif(tipo==1):
        x = open("geração_dist.csv", "a")
        x.write(nome_base + ";" + str(gen_temp) + "\n")
        x.close()
        for j in pop_temp:
            name = []
            indx = bags_temp['nome'].index(str(j))
            nm = bags_temp['inst'][indx]
            name.append(bags_temp['nome'][indx])
            name.extend(nm)

            Cpx.save_bag(name, 'bags', local + "/Bags", base_name + arquivo_de_saida, repeticao)

    elif tipo==2:
        x = open("geração_acc.csv", "a")
        x.write(nome_base + ";" + str(gen_temp) + "\n")
        x.close()
        for j in pop_temp:
            name = []
            indx = bags_temp['nome'].index(str(j))
            nm = bags_temp['inst'][indx]
            name.append(bags_temp['nome'][indx])
            name.extend(nm)
            Cpx.save_bag(name, 'bags', local + "/Bags", base_name + arquivo_de_saida, repeticao)

def max_distancia(fitness, generation=None, population=None, bags=None):
    '''

    :param fit1: fittnes 1
    :param fit2:
    :param fit3:
    :param generation: geração atual
    :param population: popoluação atual geralmente o off
    :param bags: bags atuais
    :return:
    salva em varial global a melhor populaçao de acordo com a disperção para 3 objetivos
    ideal para distancia linear
    '''
    global dist_temp, pop_temp, gen_temp, bags_temp
    if fitness[2]:
        dist_dist_media = np.mean(Cpx.dispersion(np.column_stack([fitness[0], fitness[1], fitness[2]])))
       # print(Cpx.dispersion2(np.column_stack([fitness[0], fitness[1], fitness[2]])))
    else:
        dist_dist_media = np.mean(Cpx.dispersion(np.column_stack([fitness[0], fitness[1]])))
    if dist_dist_media > dist_temp:
        dist_temp = dist_dist_media
        pop_temp = population
        gen_temp = generation
        bags_temp = bags

def max_acc(acc,generation=None, population=None, bags=None):
    '''

    :param fit1: fittnes 1
    :param fit2:
    :param fit3:
    :param generation: geração atual
    :param population: popoluação atual geralmente o off
    :param bags: bags atuais
    :return:
    salva em varial global a melhor populaçao de acordo com a disperção para 3 objetivos
    ideal para distancia linear
    '''
    global pop_temp, gen_temp, bags_temp, acc_temp

    if acc > acc_temp:
        acc_temp = acc
        pop_temp = population
        gen_temp = generation
        bags_temp = bags




tem2=[]
acc_temp=0
nome_base = 'Wine'

local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho = "/media/marcos/Data/Tese/Bases3/Bags/"
#min_score = 0

#nome_base=sys.argv[1]
#tipos=sys.argv[2]
#tipos=tipos.split(",")

#local_dataset = "/home/marcosmonteiro/Marcos/Bases3/Dataset/"
#local = "/home/marcosmonteiro/Marcos/Bases3/"
#caminho_base = "/home/marcosmonteiro/Marcos/Bases3/"
#cpx_caminho="/home/marcosmonteiro/Marcos/Bases3/Bags/"

# local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
# local = "/home/projeto/Marcos/Bases3"
# caminho_base = "/home/projeto/Marcos/Bases2/"
# cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"

########

grupo = ["overlapping", 'neighborhood', '', '', '', '']
tipos = ["F3", 'N3', '', '', '', '']

dispersao = True

fit_value1 = 1.0
fit_value2 = 1.0
fit_value3 = -1.0

nr_generation = 19
nr_individual = 100
nr_pop=100

proba_crossover = 0.99
proba_mutation = 0.01

nr_child=100
cont_crossover = 1
iteration=2
dist_temp=0

jobs = 8
parada="maxdistance"
salva_info=False


arquivo_de_saida = "marcartempo"

ini=time.time()
for t in range(1, iteration):

    off = []
    nome_individuo = 100
    repeticao = t
    generation = 0
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
        X_train, y_train, X_test, y_test, X_vali, y_vali = Cpx.routine_save_bags(local_dataset, local, nome_base,
                                                                                      repeticao)
    else:
        _, validation = Cpx.open_test_vali(local , nome_base, repeticao)
        X_vali, y_vali = Cpx.biuld_x_y(validation, X, y)
        bags = Cpx.open_bag(cpx_caminho + str(repeticao) + "/", nome_base)
    creator.create("FitnessMult", base.Fitness, weights=(fit_value1, fit_value2, fit_value3))
    creator.create("Individual", list, fitness=creator.FitnessMult)
    toolbox = base.Toolbox()
    toolbox.register("attr_item", sequencia)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                      toolbox.attr_item, 1)
    population = toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=nr_pop)

    if dispersao == True:
         distancia(primeira=True)

    toolbox.register("evaluate", fitness_dispercao_linear)
    toolbox.register("mate", cruza)
    toolbox.register("mutate", mutacao)
    toolbox.register("select", tools.selNSGA2)
    ini = time.time()
    pop = algorithms.eaMuPlusLambda(pop, toolbox, nr_child, nr_individual, proba_crossover, proba_mutation,
                                          nr_generation,
                                             generation_function=the_function)
    fim = time.time()
    print(fim - ini)