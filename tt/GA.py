import Marff as arff, newDcol, random, os
import numpy as np
from math import sqrt
from sklearn.linear_model import perceptron
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

'''
Variaveis globais

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
             # alterada por: nenhuma funcao 

dist_medias # tipo: lista
            # conteudo: distancia euclidiana media de 1 para N, (todos os bags)
            # utilizada: fitness_andre
            # atribuida: retorna_complexidades
            # alterada por: nenhuma funcao 
            
caminhos
    #abre_validacao (entrada): /media/marcos/Data/Tese/Bases/Validacao/" (repeticao) + "/Valida" + nome_base + repeticao + ".arff"
                    exe: /media/marcos/Data/Tese/Bases/Validacao/1/ValidaWine1.arff"
    
    #abre_individuos (entrada): /media/marcos/Data/Tese/AG/repeticao/geracao/Individuo" + nome_base + individuo + ".arff" 
                    ex:  /media/marcos/Data/Tese/AG/1/1/IndividuoWine1.arff"     
                    
    #cruza: (saida) "/media/marcos/Data/Tese/AG/repeticao/geracao/Individuo"+nome_base+ind
                      /media/marcos/Data/Tese/AG/1/2/IndividuoWine1.arff"  


'''



def abre_validacao():

    global X_val, y_val, nome_base, repeticao, num_classes
    v = "/media/marcos/Data/Tese/Bases/Validacao/" + str(repeticao) + "/Valida" + nome_base + str(
        repeticao) + ".arff"  # e um arquivo so de validacao por repeticao
    base_valida = arff.abre_arff(v)

    X_val, y_val = arff.retorna_instacias(base_valida)
    num_classes,*_ = arff.retorna_classes_existentes(base_valida)
    #print(X_val)
    #return X_val,y_val

def abre_individuos(individuo):
    '''
    funcao para abrir os arquivos arff de validacao e o bag
    :param individuo: int correspondente ao valor do arff ex: IndividuoWine1
    :return: X, y, todas_as_classes, base(dataset) #
    '''
    global nome_base, repeticao, num_classes, geracao
    c = "/media/marcos/Data/Tese/AG/"+str(repeticao)+"/"+str(geracao)+"/Individuo" + nome_base + str(individuo) + ".arff"#arquivo bag
    base = arff.abre_arff(c)
    X, y = arff.retorna_instacias(base)
    _, todas_as_classes,*_ = arff.retorna_classes_existentes(base)

    return X, y, base, todas_as_classes

def retorna_complexidades():
    '''
    retorna a complexidade F1 e N2 baseado no nome da base e a repeticao, altera a variavel dist_media
    :param: num_classes # numero de classes e obtido pela fucao abre arquivos
    :return: complexidades #vetor de distancias medias e vetor de todas as complexiadades
    '''
    global nome_base, repeticao, num_classes, dist_medias, geracao
    complexidades = list()
    for i in range(1, 101):
        c = "/media/marcos/Data/Tese/AG/"+str(repeticao)+"/" + str(geracao) + "/Individuo" + nome_base + str(i) + ".arff"
        F1, N2, *_ = newDcol.retorna_complexidade(c, complexidades="-F 1 -N 2", num_classes=num_classes, media=True)
        cpx = [F1, N2]
        complexidades.append(cpx)
    #print(complexidades)
    dist_medias = list()
    for j in range(len(complexidades)):
        dist=0
        for l in range(len(complexidades)):
            if(j==l):
                continue
            else:
                a=complexidades[j]
                b=complexidades [l]
                dist+=sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
        dist_medias.append(dist/100)
    #print(dist_medias)
    #dist_media=[1.1830971486614692, 0.8701683484434686, 1.4163743022955613, 0.8183401459794912, 0.8200386502626208, 1.0342248580289428, \
    # 1.0145519073703961, 2.116491995970377, 0.907030667179295, 0.9001307721983031, 1.1840208486505817, 0.8251017188798133, 0.8683221037543006, \
    # 1.3150538965271028, 0.8489837849248356, 0.9486167171393383, 0.8840543262846411, 0.8979178204824635, 1.6892691841552565, 0.8276352762966465, \
    # 1.547701309397448, 0.9733463495991246, 0.8891515872430366, 0.9355194339630064, 0.8335670373293764, 0.8712145856049536, 0.8634810416188492, \
    # 0.949716572100299, 1.0881027607331837, 0.8578750089757495, 1.019327827904733, 0.8608728723070285, 1.6170189302274827, 3.2381608040838774, \
    # 1.0584364316272141, 2.0166212985713696, 1.01678312846505, 0.8514709164030931, 1.350270780005099, 0.9673181849750117, 1.0644872063883188, \
    # 2.5150702139747976, 1.9531638427288756, 0.8561420228094818, 0.8422294426682467, 1.584237464102468, 0.8226035152659947, 1.0165654864644456, \
    # 1.0010439240452238, 0.849519400966241, 0.8227576995660152, 1.308675137495249, 1.3847386625636888, 0.9190302992900673, 0.820924898879569, \
    # 1.3782136609505298, 1.0721165629704759, 0.8389385775449177, 0.9367351991730136, 2.0898710267953104, 0.8293297084431067, 0.9484400868923086, \
    # 0.8188521734879706, 0.8557938695282417, 1.2400333681168403, 1.151351026131332, 2.635226691273881, 1.0682109688023667, 1.2950189419476263, \
    # 1.3945240899964682, 1.921366580475652, 0.8631874037358902, 0.9856432143423991, 0.9567001306480691, 1.1747657267826852, 0.8562586117454011, \
    # 0.8287674664225422, 0.9447769228339854, 1.3956452894455875, 1.8403407817070674, 1.3713394669800931, 1.4494048663871286, 2.3689885536427067, \
    # 0.8978954960181267, 1.000480901727328, 1.5754892512310223, 0.8927853213839654, 0.869396530824599, 0.8441296859447559, 0.8694673928275123, \
    # 0.9344194078654178, 0.9109440465143007, 0.8201213060144464, 1.0997634488669936, 1.2007643792276435, 1.5144590185123088, 0.857952361592603, 1.4102564623403109, 0.8213497260874556, 0.8776029666123896]
    return complexidades

def fitness_andre(individuo):
    '''
    Funcao de fitness, retorna a acuracia e a distancia do bag requerido
    :param individuo: tipo int #será passado por parametro para gerar o nome do arquivo
    :return: perc.score(X_val, y_val)+ dist_medias[arquivo]
    '''
    global dist_medias, X_val, y_val

    X, y,*_= abre_individuos(individuo)
    perc = perceptron.Perceptron()
    perc.fit(X, y)
    #print(dist_medias[individuo])
    return perc.score(X_val, y_val)+ dist_medias[individuo]

def fitness_moga(individuo):
    '''
    Funcao de fitness, retorna a acuracia e a distancia do bag requerido
    :param individuo: tipo int #será passado por parametro para gerar o nome do arquivo
    :return: perc.score(X_val, y_val), dist_medias[arquivo]
    '''
    global dist_medias, X_val, y_val

    X, y,*_= abre_individuos(individuo)
    perc = perceptron.Perceptron()
    perc.fit(X, y)

    return perc.score(X_val, y_val), dist_medias[individuo]

def cruza(ind, ind2):
    global nome_base, geracao, repeticao
    #random.seed(64)
    inicio=0
    fim=0
    #diferenca=0
    X3=dict()
    X4 = dict()
    X3['data']= list()
    X4['data'] = list()
    X,y,base, todas_as_classes=abre_individuos(ind)
    X2,y2,*_=abre_individuos(ind2)

    while(y[inicio]==y[fim]):
        inicio=random.randint(0, len(y)-1)
        fim = random.randint(inicio, len(y) - 1)
        #diferenca=fim-inicio
    for i in range(len(X)):
        if(i<=inicio or i>=fim):
            X3['data'].append(X[i])
            X4['data'].append(X2[i])
        else:
            X3['data'].append(X2[i])
            X4['data'].append(X[i])

    for j in range(len(y)):
        X3['data'][j].append(y[j])
        X4['data'][j].append(y[j])
    gera=1+geracao
    pasta = "/media/marcos/Data/Tese/AG/" + str(repeticao) + "/" + str(gera)
    if (os.path.exists(pasta) == False):
        os.system("mkdir -p "+pasta)

    nome="/Individuo"+nome_base+str(ind)
    nome2 = "/Individuo"+nome_base + str(ind2)
    arff.cria_arff(base,X3,todas_as_classes,pasta,nome)
    arff.cria_arff(base,X4,todas_as_classes,pasta,nome2)

def mutacao(individuo):

    global geracao
    X=dict()
    X['data'], y, base, todas_as_classes = abre_individuos(individuo)
    #print(X['data'])
    inst = 0
    inst2 = len(y)
    individuo2 = random.randint(1, 100)
    X2, *_ = abre_individuos(individuo2)
    while y[inst] != y[inst2-1]:
        inst = random.randint(0, len(y)-1)
        inst2 = random.randint(0, len(y) - 1)

    gera=1+geracao
    pasta = "/media/marcos/Data/Tese/AG/" + str(repeticao) + "/" + str(gera)
    print(pasta)
    for j in range(len(y)):
        X['data'][j].append(y[j])
    nome = "/Individuo" + nome_base + str(individuo)
    arff.cria_arff(base, X, todas_as_classes, pasta, nome)





nome_base='Wine'
repeticao=1
geracao=1
num_classes=2

dist_medias=[]
X_val=[]
y_val=[]


abre_validacao()
_=retorna_complexidades()
#dist=fitness_andre(1)
#cruza(1,2)
#mutacao(1)
#print(geracao)
#fitness_andre(caminho, valida_caminho, nome_base)

#####################################################################################################################################
#X,*_=abre_individuos(1)
individual_size = 1
nr_generation = 2
qt_selection = 6 # (elitismo)
nr_children_generation = 100
proba_crossover = 0.8
proba_mutation = 0.01
current_ind = 1



creator.create("Fitness", base.Fitness, weights=(1.0))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()

toolbox.register("attr_item", random.randint, 1, 100)# virar 123....

toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_item, 1)
ind1 = toolbox.individual()
print(ind1)
exit(0)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_andre)
toolbox.register("mate", cruza)
toolbox.register("mutate", mutacao)
toolbox.register("select", tools.selNSGA2)
#toolbox.register("select", tools.selRoulette)


pop = toolbox.population(n=qt_selection)

hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)


algorithms.eaMuPlusLambda(pop, toolbox, qt_selection, nr_children_generation, proba_crossover, proba_mutation, nr_generation, stats,
                          halloffame=hof, verbose=True)
#

#
#
#
# for i in range(len(hof)):
#     print("Accuracy {}: {} {}".format(i, CheckAccuracy(X_test, Y_test, hof[i]), hof[i]))
#     sys.stdout.flush()
# #
# classif_list = [1 for i in range(individual_size)]
# print("Accuracy all: {} {}".format(CheckAccuracy(X_test, Y_test, classif_list), classif_list))






















#################################################################################
#funcao sem uso

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