import Marff as arff, newDcol, random, os, shutil, sys
from math import sqrt
from sklearn.linear_model import perceptron
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.preprocessing import MinMaxScaler
import complexity_pcol as dcol
import numpy as np

def altera_arquivo_marcelo():
    '''
    da nome aos bags nesse caso 1 a 100
    :return:
    '''
    global repeticao, nome_base, geracao
    arq = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base +str(geracao)+ ".indx")
    arqtemp = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + ".indxTemp", 'w')
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
    os.system("cp -r /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + str(geracao)+ ".indxTemp /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base +str(geracao)+ ".indx")
    os.system("rm /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + str(geracao) +".indxTemp")



def abre_arquivo(individuo=None, valida=False):
    global nome_base, repeticao, geracao
    if individuo:
        arq=open("/media/marcos/Data/Tese/GA2/"+str(repeticao)+"/"+nome_base+str(geracao)+".indx")
        for i in arq:
            texto=i
            #print(str(individuo))
            if(str(individuo)==texto.split(" ")[0]):
                indx_bag=texto.split(" ")
                arq.close()
                indx_bag=indx_bag[1:]
                break
    elif valida:
        arq = open("/media/marcos/Data/Tese/Bases2/Validacao/" + str(repeticao) + "/" + nome_base+".idx")
        texto=arq.readline()
        indx_bag=texto.split(" ")
        arq.close()
    #print(indx_bag)
    return indx_bag

def monta_arquivo(indx_bag,vet_class=False):
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :return:
    '''
    #print(indx_bag)
    global nome_base, classes

    #print(indx_bag)
    X_data=[]
    y_data=[]
    arq2=("/media/marcos/Data/Tese/Bases2/Dataset/"+nome_base+".arff")
    arq3=arff.abre_arff(arq2)
    X,y=arff.retorna_instacias(arq3)
    if(vet_class):
        _,classes,*_=arff.retorna_classes_existentes(arq3)
    for i in indx_bag:
        #print(int(i))
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    #print(X_data)
    #exit(0)
    return X_data, y_data

def cruza(ind1, ind2):
    '''
    Para funcionar os bags devem ter o mesmo tamanho
    :param ind1:
    :param ind2:
    :return:
    '''
    global nome_individuo, repeticao, nome_base, geracao
    print("Cruzamento", ind1,ind2)
    #print(ind1, ind2, geracao)
    individuo_arq = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + str(geracao)+".indx", 'a')
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

    individuo_arq.close()

    return creator.Individual(ind1), creator.Individual(ind2)



def mutacao(ind):
    global geracao, off, nome_individuo, repeticao
    print("mutacao", ind)
    individuo_arq = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + str(geracao) + ".indx",
                         'a')
    indx_bag1 = abre_arquivo(ind[0])
    X,y_data=monta_arquivo(indx_bag1)
    ind_out=[]
    ind_out.append(str(nome_individuo))
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

    for j in ind_out:
        if (j != ind_out[-1]):
            individuo_arq.write(j)
            individuo_arq.write(" ")
        else:
            individuo_arq.write(j)
    ind[0] = nome_individuo
    nome_individuo += 1
    return ind,

def fitness_f1_n2(individuo):
    global classes

    print("fitnes", individuo)

    indx_individuo=abre_arquivo(individuo[0])
    X_data,y_data=monta_arquivo(indx_individuo)
    #print(X_data)
    #print(y_data)

    scaler = MinMaxScaler()
    scaler.fit(X_data)
    transformed_data=scaler.transform(X_data)
    #print(transformed_data)

    complex=dcol.PPcol(classes=classes)
    complexidades=complex.xy_measures(transformed_data,y_data)
    F1 = np.average(complexidades['F1'])
    N2 = np.average(complexidades['N2'])
    perc=perceptron.Perceptron()
    perc.fit(X_data,y_data)
    score = perc.score(X_valida,y_valida)
    return F1, N2, score,

def sequencia():
    global seq
    seq += 1
    return seq

def the_function(population, gen, offspring):
    '''
    responsavel por alterar a geração, assim como zerar variaveis, alterar populaçoes, e copiar arquivos
    :param population: populacao, retorna do DEAP
    :param gen: geração Retorna do DEAP
    :param offspring: nova população
    :return:
    '''
    global geracao, off, X_valida, y_valida
    print("the_fuction", (population))
    off=[]
    geracao = gen
    geracao_arq = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + str(geracao) + ".indx",
                         'a')
    arq = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + str(geracao-1) + ".indx")
    for i in range(len(population)):
        off.append(population[i][0])
    # if (geracao==30):
    #     csv.write('{};{};{};'.format(nome_base,str(repeticao),str(geracao)))
    #     for i in range(len(population)):
    #         off.append(population[i][0])
    #         if (population[i][0] == population[-1][0]):
    #             csv.write(str(population[i][0]) + '\n')
    #         else:
    #             csv.write(str(population[i][0]) + ';')
    # if (os.path.exists(pasta2) == False):
    #     os.system("mkdir -p " + pasta2)
    # for i in population:
    #     shutil.copy2(pasta + "/Individuo" + nome_base + str(i[0]) + '.arff', pasta2)
    # abre_validacao()
    # _ = retorna_complexidades(population=population)
    for j in population:
        for i in arq:
            texto = i
            print(texto)
            if (str(j[0]) == texto.split(" ")[0]):
                print(i)
                geracao_arq.write(i)

    indx_valida = abre_arquivo(valida=True)
    X_valida, y_valida = monta_arquivo(indx_valida)


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
    #print(off)
    return off

nome_individuo=101
nome_base="Wine"
repeticao=1
geracao=0
seq=0
classes=[]
off=[]
indx_valida=abre_arquivo(valida=True)
X_valida,y_valida=monta_arquivo(indx_valida,vet_class=True)





nr_generation = 30
proba_crossover = 0.99
proba_mutation = 0.01
# current_ind = 14
fit_value1=1.0
fit_value2=1.0
fit_value3=1.0
#valor=1



creator.create("Fitness", base.Fitness, weights=(fit_value1,fit_value2,fit_value3))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()

toolbox.register("attr_item", sequencia)

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, 1)


population=toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n=100)
toolbox.register("evaluate", fitness_f1_n2)
toolbox.register("mate", cruza)
toolbox.register("mutate", mutacao)
toolbox.register("select", tools.selSPEA2 )
algorithms.eaMuPlusLambda(pop, toolbox, 100, 100, proba_crossover, proba_mutation, nr_generation, generation_function=the_function, popu=populacao)

#print(classes)
#fitness_f1_n2([5])
#print(y_valida)
#altera_arquivo_marcelo()

#cruza([1],[2])
#nome_bag()
#print(x)

