import Marff as arff, newDcol, random, os
from math import sqrt
from sklearn.linear_model import perceptron
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

'''
Variaveis globais
caminho_todas # tipo string ->"/media/marcos/Data/Tese/AG/"
            #conteudo: camnho para pastas
            #utilizada: retorna_complexidades, abre_individuos, cruza, mutacao
            #atribuida: (fora)
            #modificada: nenhuma.

caminho_valida # tipo string ->"/media/marcos/Data/Tese/Bases/Validacao/"
            #conteudo: camnho para pastas
            #utilizada: retorna_validacao
            #atribuida: (fora)
            #modificada: nenhuma.         

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

geracao     # tipo: int
            # conteudo: geracao(destino pasta)
            # utilizada em: retorna_complexidades, abre_individuos, cruza, mutacao
            # atribuida: (fora)
            # alterada por: nenhuma

num_classes # tipo: int
            # conteudo: numero de classes da base em uso
            # utilizada em: retorna_complexidades
            # atribuida: abre_validacao
            # alterada por: nenhuma funcao 

X_val, y_val # tipo: lista de lista, lista
             # conteudo: instancias, classes
             # utilizada: fitness_andre
             # atribuida: abre_validacao
             # alterada por: The_function

dist_medias # tipo: lista
            # conteudo: distancia euclidiana media de 1 para N, (todos os bags)
            # utilizada: fitness_andre
            # atribuida: retorna_complexidades
            # alterada por: nenhuma funcao 

n           #tipo: inteiro
            #conteudo: nome aos arquivos
            #utilizada: the_fuction, mutacao, cruza
            #atribuida: fora, the_funcion
            #alterada: mutacao, thefucntion, cruza

caminhos
    #abre_validacao (entrada): /media/marcos/Data/Tese/Bases/Validacao/" (repeticao) + "/Valida" + nome_base + repeticao + ".arff"
                    exe: /media/marcos/Data/Tese/Bases/Validacao/1/ValidaWine1.arff"

    #abre_individuos (entrada): /media/marcos/Data/Tese/AG/repeticao/geracao/Individuo" + nome_base + individuo + ".arff" 
                    ex:  /media/marcos/Data/Tese/AG/1/1/IndividuoWine1.arff"     

    #cruza: (saida) "/media/marcos/Data/Tese/AG/repeticao/geracao/Individuo"+nome_base+ind
                      /media/marcos/Data/Tese/AG/1/2/IndividuoWine1.arff"  


'''


def abre_validacao():
    global X_val, y_val, nome_base, repeticao, num_classes, caminho_valida
    v = caminho_valida + str(repeticao) + "/Valida" + nome_base + str(
        repeticao) + ".arff"  # e um arquivo so de validacao por repeticao
    base_valida = arff.abre_arff(v)

    X_val, y_val = arff.retorna_instacias(base_valida)
    num_classes, *_ = arff.retorna_classes_existentes(base_valida)
    # print(y_val)
    # return X_val,y_val


def abre_individuos(individuo):
    '''
    funcao para abrir os arquivos arff de validacao e o bag
    :param individuo: int correspondente ao valor do arff ex: IndividuoWine1
    :return: X, y, todas_as_classes, base(dataset) #
    '''
    global nome_base, repeticao, num_classes, geracao
    c = caminho_todas + str(repeticao) + "/" + str(geracao) + "/Individuo" + nome_base + str(
        individuo) + ".arff"  # arquivo bag
    base = arff.abre_arff(c)
    X, y = arff.retorna_instacias(base)
    _, todas_as_classes, *_ = arff.retorna_classes_existentes(base)

    return X, y, base, todas_as_classes


def retorna_complexidades():
    '''
    retorna a complexidade F1 e N2 baseado no nome da base e a repeticao, altera a variavel dist_media
    :param: num_classes # numero de classes e obtido pela fucao abre arquivos
    :return: complexidades #vetor de distancias medias e vetor de todas as complexiadades
    '''
    global nome_base, repeticao, num_classes, dist_medias, geracao, caminho_todas
    complexidades = list()
    for i in range(1, 101):
        c = caminho_todas + str(repeticao) + "/" + str(geracao) + "/Individuo" + nome_base + str(i) + ".arff"
        F1, N2, *_ = newDcol.retorna_complexidade(c, complexidades="-F 1 -N 2", num_classes=num_classes, media=True)
        cpx = [F1, N2]
        complexidades.append(cpx)
    # print(complexidades)
    dist_medias = list()
    for j in range(len(complexidades)):
        dist = 0
        for l in range(len(complexidades)):
            if (j == l):
                continue
            else:
                a = complexidades[j]
                b = complexidades[l]
                dist += sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
        dist_medias.append(dist / 100)
    # print(dist_medias)

    return complexidades


def fitness_andre(individuo):
    ind = individuo[0]
    #print(individuo)

    '''
    Funcao de fitness, retorna a acuracia e a distancia do bag requerido
    :param individuo: tipo int #será passado por parametro para gerar o nome do arquivo
    :return: perc.score(X_val, y_val)+ dist_medias[arquivo]
    '''
    global dist_medias, X_val, y_val

    X, y, *_ = abre_individuos(ind)
    perc = perceptron.Perceptron()
    perc.fit(X, y)
    #print(len(dist_medias))
    out = float(perc.score(X_val, y_val) + dist_medias[ind-1])
    # print(out)
    return out,


def fitness_moga(individuo):
    indi = individuo[0]
    '''
    Funcao de fitness, retorna a acuracia e a distancia do bag requerido
    :param individuo: tipo int #será passado por parametro para gerar o nome do arquivo
    :return: perc.score(X_val, y_val), dist_medias[arquivo]
    '''
    global dist_medias, X_val, y_val

    X, y, *_ = abre_individuos(indi)
    perc = perceptron.Perceptron()
    perc.fit(X, y)

    out = float(perc.score(X_val, y_val))
    out2=float(dist_medias[indi-1])
    return out, out2,


def cruza(indi, indi2):

    print("cruzamento entre: individuo1 {} e individuo2 {}".format(indi,indi2))

    #exit(0)
    global nome_base, geracao, repeticao, caminho_todas, n
    # random.seed(64)

    inicio = 0
    fim = 0
    # diferenca=0
    X3 = dict()
    X4 = dict()
    X3['data'] = list()
    X4['data'] = list()
    X, y, base, todas_as_classes = abre_individuos(indi[0])
    X2, y2, *_ = abre_individuos(indi2[0])

    while (y[inicio] == y[fim]):
        inicio = random.randint(0, len(y) - 1)
        fim = random.randint(inicio, len(y) - 1)
        # diferenca=fim-inicio
    for i in range(len(X)):
        if (i <= inicio or i >= fim):
            X3['data'].append(X[i])
            X4['data'].append(X2[i])
        else:
            X3['data'].append(X2[i])
            X4['data'].append(X[i])

    for j in range(len(y)):
        X3['data'][j].append(y[j])
        X4['data'][j].append(y[j])

    pasta = caminho_todas + str(repeticao) + "/" + str(geracao+1)
    if (os.path.exists(pasta) == False):
        os.system("mkdir -p " + pasta)

    nome = "/Individuo" + nome_base + str(n)

    indi[0]=n
    n = n + 1
    #nome2 = "/Individuo" + nome_base + str(n)
    arff.cria_arff(base, X3, todas_as_classes, pasta, nome)
    #arff.cria_arff(base, X4, todas_as_classes, pasta, nome2)
    k=n+1
    indi2[0]=k

    return creator.Individual(indi), creator.Individual(indi2)


def mutacao(individuo):
    ind = individuo[0]

    global geracao, caminho_todas, n
    X = dict()
    X['data'], y, base, todas_as_classes = abre_individuos(ind)
    # print(X['data'])
    inst = 0
    inst2 = len(y)
    ind2 = random.randint(1, 100)
    X2, *_ = abre_individuos(ind2)
    while y[inst] != y[inst2 - 1]:
        inst = random.randint(0, len(y) - 1)
        inst2 = random.randint(0, len(y) - 1)
    pasta = caminho_todas + str(repeticao) + "/" + str(geracao+1)
    print('mutacaooooo', individuo, ind2)
    for j in range(len(y)):
        X['data'][j].append(y[j])
    nome = "/Individuo" + nome_base + str(n)
    arff.cria_arff(base, X, todas_as_classes, pasta, nome)
    individuo[0] = n
    n=n+1
    return individuo,


caminho_todas = "/media/marcos/Data/Tese/AG/"
caminho_valida = "/media/marcos/Data/Tese/Bases/Validacao/"
nome_base = 'Wine'
repeticao = 1
geracao = 1

num_classes = 2

dist_medias = []
X_val = []
y_val = []

abre_validacao()
_ = retorna_complexidades()
n=1

#####################################################################################################################################

verbose = True

nr_generation = 30
proba_crossover = 0.99
proba_mutation = 0.01
# current_ind = 14



creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()

seq = 0


def sequencia():
    global seq
    seq += 1
    #print(seq)
    return seq

toolbox.register("attr_item", sequencia)

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, 1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_andre)
toolbox.register("mate", cruza)
toolbox.register("mutate", mutacao)
toolbox.register("select", tools.selBest)
# toolbox.register("select", tools.selRoulette)


pop = toolbox.population(n=100)

hof = tools.ParetoFront()
# stats = tools.Statistics(lambda ind: ind.fitness.values)
#
# stats.register("avg", np.mean, axis=0)
# stats.register("std", np.std, axis=0)
# stats.register("min", np.min, axis=0)
# stats.register("max", np.max, axis=0)

# sys.stdout.flush()


def the_function(population, gen, offspring):
    global geracao, dist_medias, X_val, y_val, n
    pasta = caminho_todas + str(repeticao) + "/" + str(geracao)
    pasta2 = caminho_todas + str(repeticao) + "/" + str(geracao+1)
    dist_medias = []
    X_val = []
    y_val = []

    geracao = gen
    print('Eletismo: ',population)
    print('Populacao: ', offspring)
    print(len(offspring))
    for i in population:
       os.system("cp "+pasta+"/Individuo" + nome_base + str(i[0]) + '.arff '+pasta2+"/Individuo" + nome_base + str(n)+'.arff')
       n=n+1
    n = 1
    #exit(0)

    abre_validacao()
    _ = retorna_complexidades()


algorithms.eaMuCommaLambda(pop, toolbox, 4, 96, proba_crossover, proba_mutation, nr_generation, generation_function=the_function)






#




#################################################################################
# funcao sem uso

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