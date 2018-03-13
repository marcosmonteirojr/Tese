from deap.tools.selection import selBest

import Marff as arff, newDcol, random, os, shutil
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
            # alterada por: nenhuma, 

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

dist        # tipo: dicinario
            # conteudo: distancia euclidiana media de 1 para N, (todos os bags), nome do individuo testado
            # utilizada: fitness_andre
            # atribuida: retorna_complexidades
            # alterada por: nenhuma funcao 

n           #tipo: inteiro
            #conteudo: nome aos arquivos
            #utilizada: the_fuction, mutacao, cruza
            #atribuida: fora, the_funcion
            #alterada: mutacao, thefucntion, cruza

contador_cruzamento #tipo: inteiro
                    #conteudo: contador para chamar a funcao retorna complexidade
                    #utilizada: em cruza, mutacao, retorna_complexidade
                    #atribuida: fora
                     #alterada: mutacao, cruza

off          #tipo: lista de lista
             #conteudo: hora populacao, hora populacao+offspring
             #utilizada: mutacao, thefunction
             #atribuida: populacao
             #alterada: populacao, thefunction                 
   
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


def retorna_complexidades(population=None, primeira=None):
    '''
    retorna a complexidade F1 e N2 baseado no nome da base e a repeticao, altera a variavel dist
    possui tres situacoes, 1 primeira geracao, demais populacao (nova populacao, antes da geracao), e populacao geracao (populacao
    de uma nova geracao)
    :param: populacao, e primeira geracao
    :return: complexidades #vetor de distancias medias e vetor de todas as complexiadades, e variavel global dist

    '''

    global nome_base, repeticao, num_classes, geracao, caminho_todas, pop, contador_complexidades, n, dist
    complexidades = list()

    print('##########################-Complexidades-##############################################')
    if (geracao == 0 and primeira==True):
        dist = dict()
        dist['nome'] = list()
        dist['dist'] = list()
        dist['nome'] = pop
        print('primeira populacao', dist['nome'])
        for i in dist['nome']:
            c = caminho_todas + str(repeticao) + "/" + str(geracao) + "/Individuo" + nome_base + str(i[0]) + ".arff"
            F1, N2, *_ = newDcol.retorna_complexidade(c, complexidades="-F 1 -N 2", num_classes=num_classes, media=True)
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
        print('geracao de populacao', dist['nome'])
        for i in dist['nome']:
            # print(i[0])
            c = caminho_todas + str(repeticao) + "/" + str(geracao) + "/Individuo" + nome_base + str(i[0]) + ".arff"
            F1, N2, *_ = newDcol.retorna_complexidade(c, complexidades="-F 1 -N 2", num_classes=num_classes, media=True)
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
        inicio=n-100
        for i in range(inicio,n):
            x=[]
            x.append(i)
            dist['nome'].append(x)
        print('demais populacao', dist['nome'])
        complexidades = list()
        for i in dist['nome']:
            # print(i[0])
            c = caminho_todas + str(repeticao) + "/" + str(geracao) + "/Individuo" + nome_base + str(i[0]) + ".arff"
            F1, N2, *_ = newDcol.retorna_complexidade(c, complexidades="-F 1 -N 2", num_classes=num_classes,
                                                      media=True)
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
        #print('complexidade', dist['dist'])
    return complexidades



def fitness_andre(individuo):
    '''
    Funcao de fitness, retorna a acuracia e a distancia do bag requerido
    :param individuo: tipo int #será passado por parametro para gerar o nome do arquivo
    :return: perc.score(X_val, y_val)+ dist_medias[arquivo]
    '''
    global dist, X_val, y_val
    #print('fitness')
    distancia = 0

    ind = individuo[0]
    for i in range(len(dist['nome'])):

        if dist['nome'][i][0] == ind:
            distancia = dist['dist'][i]
            #print('nome, distancia', dist['nome'][i][0], distancia)
            break
    X, y, *_ = abre_individuos(ind)
    perc = perceptron.Perceptron()
    perc.fit(X, y)
    #print(len(dist_medias))
    out = float(perc.score(X_val, y_val) + distancia)

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
    out2 = float(dist_medias[indi-1])
    return out, out2,


def cruza(indi, indi2):
    global nome_base, geracao, repeticao, caminho_todas, n, contador_cruzamento, population
    print("cruzamento entre: individuo1 {} e individuo2 {} e contador {}".format(indi,indi2,contador_cruzamento))
    inicio = 0
    fim = 0
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
    pasta = caminho_todas + str(repeticao) + "/" + str(geracao)
    if (os.path.exists(pasta) == False):
        os.system("mkdir -p " + pasta)
    nome = "/Individuo" + nome_base + str(n)
    indi[0]=n
    indi2[0] = n
    arff.cria_arff(base, X3, todas_as_classes, pasta, nome)
    n = n + 1
    contador_cruzamento = contador_cruzamento + 1
    if(contador_cruzamento==101):
        contador_cruzamento=1
        _=retorna_complexidades()
    return creator.Individual(indi), creator.Individual(indi2)

def mutacao(individuo):
    global geracao, caminho_todas, n, contador_cruzamento, off
    ind = individuo[0]
    X = dict()
    X['data'], y, base, todas_as_classes = abre_individuos(ind)
    print('OFF, tamanho', len(off), off )
    inst = 0
    inst2 = len(y)
    if(geracao==0 and off==[]):
        ind2 = random.randint(1, 100)
    else:
        ind2=random.sample(off,1)
        ind2=ind2[0]
    print('mutacaooooo e contador', individuo, ind2, contador_cruzamento)
    X2, *_ = abre_individuos(ind2)
    while y[inst] != y[inst2 - 1]:
        inst = random.randint(0, len(y) - 1)
        inst2 = random.randint(0, len(y) - 1)
    pasta = caminho_todas + str(repeticao) + "/" + str(geracao)
    for j in range(len(y)):
        X['data'][j].append(y[j])
    nome = "/Individuo" + nome_base + str(n)
    arff.cria_arff(base, X, todas_as_classes, pasta, nome)
    individuo[0] = n
    n=n+1
    contador_cruzamento=contador_cruzamento+1
    if (contador_cruzamento == 101):
        contador_cruzamento=1
        _ = retorna_complexidades()
    return individuo,

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
    global X_val,y_val, n, geracao, off
    pasta = caminho_todas + str(repeticao) + "/" + str(geracao)
    pasta2 = caminho_todas + str(repeticao) + "/" + str(geracao + 1)
    X_val = []
    y_val = []
    off=[]
    for i in population:
        j=([i][0])
        j=j[0]
        off.append(j)
    geracao = gen
    if (os.path.exists(pasta2) == False):
        os.system("mkdir -p " + pasta2)
    population = sorted(population)
    for i in population:
        shutil.copy2(pasta + "/Individuo" + nome_base + str(i[0]) + '.arff', pasta2)
    abre_validacao()
    _ = retorna_complexidades(population=population)

def populacao(populacao_total):
    '''
    callback para offspring
    :param populacao_total: offspring DEAP
    :return: off (offspring+populacao)
    '''
    global off
    off=[]
    for i in populacao_total:
        j=([i][0])
        j=j[0]
        off.append(j)
    return off

caminho_todas = "/media/marcos/Data/Tese/AG/"
caminho_valida = "/media/marcos/Data/Tese/Bases/Validacao/"
nome_base = 'Wine'
repeticao = 1
geracao = 0
off=[]
num_classes = 2
dist = dict()
dist['nome'] = list()
dist['dist'] = list()
dist_medias = []
X_val = []
y_val = []
fit_andre=[]
seq = 0
contador_complexidades=0
contador_cruzamento=1
n=101

#####################################################################################################################################

verbose = True

nr_generation = 30
proba_crossover = 0.99
proba_mutation = 0.01
# current_ind = 14



creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()

toolbox.register("attr_item", sequencia)

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, 1)

population=toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_andre)
toolbox.register("mate", cruza)
toolbox.register("mutate", mutacao)
toolbox.register("select", tools.selBest )
# toolbox.register("select", tools.selRoulette)




pop = toolbox.population(n=100)
abre_validacao()
_ = retorna_complexidades(primeira=True)

algorithms.eaMuPlusLambda(pop, toolbox, 100, 100, proba_crossover, proba_mutation, nr_generation, generation_function=the_function, popu=populacao)






