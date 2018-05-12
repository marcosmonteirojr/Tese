import Marff as arff, newDcol, random, os, shutil, sys
from math import sqrt
from sklearn.linear_model import perceptron
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

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



def abre_arquivo(individuo=None, valida=None):
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
    return indx_bag

def monta_arquivo(indx_bag):
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :return:
    '''
    #print(indx_bag)
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
    return ind

def fitness_f1_n2(individuo):
    indx_indivudo=abre_arquivo(individuo)
    X_data,y_data=monta_arquivo(indx_indivudo)



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
            # print(str(individuo))
            if (str(j[0]) == texto.split(" ")[0]):
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
indx_valida=abre_arquivo(valida=True)
X_valida,y_valida=monta_arquivo(indx_valida)
print(y_valida)
#altera_arquivo_marcelo()

#cruza([1],[2])
#nome_bag()
#print(x)

