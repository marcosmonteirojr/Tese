# import arff, numpy, random, os
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.externals import joblib
# from sklearn.model_selection import GridSearchCV
# global resultados, media, maior
# from deap import algorithms
# from deap import base
# from deap import creator
# from deap import tools
# from sklearn.ensemble import VotingClassifier
# classifiers = list()
# resultados=[]
# current_ind=1
#
#
# # def acuracia(repeticao, X_test, y_test):#mede a acuracia do classificador, retorna a media, o maior, os resultados e o nome do maior
# #     resultados = []
# #     anterior=0
# #     for i in range(repeticao):
# #         classificador = joblib.load('clf/TreeClas' + str(i) + '.pkl')
# #         atual=classificador.score(X_test,y_test)
# #         resultados.append(atual)
# #         if (atual>anterior):
# #             anterior=atual
# #             nome='TreeClas'+str(i)
# #     maior=max(resultados)
# #     media=numpy.mean(resultados)
# #     return maior, media, resultados, nome
#
# # def check_marcos(individual):
# #
# #     c=[]
# #     n=[]
# #
# #     for j in range(len(individual)):
# #         if (individual[j] == 1):
# #             nome_val=nome[j]
# #             pred_val = classifiers[j]
# #             c.append(pred_val)
# #             n.append(nome_val)
# #
# #     eclf = VotingClassifier(estimators=zip(n,c),voting='hard')
# #     c.append(eclf)
# #     n.append('Ensemble')
# #     scores=[]
# #   #  print(zip(c,n))
# #     for clf, label in zip(c,n):
# #         scores.append( cross_val_score(clf, X_test, Y_test, cv=2, scoring='accuracy'))
# #     return np.mean(scores),
#
# # def evalEnsemble(individual):
# #    # print (individual)
# #     global current_ind
# #     c=[]
# #     n=[]
# #
# #     for j in range(len(individual)):
# #         if (individual[j] == 1):
# #             nome_val=nome[j]
# #             pred_val = classifiers[j]
# #             c.append(pred_val)
# #             n.append(nome_val)
# #
# #     eclf = VotingClassifier(estimators=zip(n,c),voting='hard')
# #     c.append(eclf)
# #     n.append('Ensemble')
# #     scores=[]
# #   #  print(zip(c,n))
# #     for clf, label in zip(c,n):
# #         #scores.append( cross_val_score(clf, X_val, Y_val, cv=2, scoring='accuracy'))
# #
# #     print("Accuracy: %f (+/- %0.2f) [%s]" % (np.mean(scores), np.std(scores), label))
# #     #print(scores.mean())
# #
# #     return np.mean(scores),
#
# def cria_dataset(dataset): #cria treino teste e validacao e grava em arquivo
#     print('Criando dataset\n')
#     labels =[]
#     instancias=[]
#     for i in range(len(dataset['data'])):  # percorre a base e separa os labels das classes
#         #labels.append(dataset['data'][i][16])
#         #print (labels)
#         #labels = [str(w) for w in labels]#transforma as classes em string
#         labels.append(ord(dataset['data'][i][-1])-65)
#
#
#     for i in dataset['data']:
#         instancias.append(i[:-1])#salva so as instancias(sem classes)
#
#     #tree = DecisionTreeClassifier()
#     X_train, X_x, y_train, y_y = train_test_split(instancias, labels, test_size=0.5, random_state=True, stratify=labels)#divide a base de treino
#     X_test, X_val, y_test, y_val= train_test_split(X_x, y_y, test_size=0.5, random_state=None, stratify=y_y)#divide a base entre teste e validacao
#     # tree.fit(X_train,y_train)
#     # print(tree.score(X_val,y_val))
#
#     dados = dict()
#     X=list()
#     for i in range(len(y_train)):
#         linha_arff=list()
#         for j in X_train[i]:
#             linha_arff.append(j)
#         linha_arff.append(y_train[i])
#         X.append(linha_arff)
#     dados['data'] = X
#     print ('Criando treino\n')
#     cria_arff(dataset,dados,"treino",'dataset')
#
#
#     dados = dict()
#     X = list()
#     for i in range(len(y_test)):
#         linha_arff = list()
#         for j in X_test[i]:
#             linha_arff.append(j)
#         linha_arff.append(y_test[i])
#         X.append(linha_arff)
#     dados['data'] = X
#     print ('Criando teste\n')
#     cria_arff(dataset, dados, "teste",'dataset')
#
#
#     for i in range(len(y_val)):
#         linha_arff = list()
#         for j in X_val[i]:
#             linha_arff.append(j)
#         linha_arff.append(y_val[i])
#         X.append(linha_arff)
#     dados['data'] = X
#     cria_arff(dataset, dados, "validacao",'dataset')
#     print ('Criando validacao\n')
#     return X_train, X_test, X_val, y_train, y_test, y_val # X_ -> bases sem labels, y_-> labels
#
#
# def cria_arff(info, data, nome, pasta):
#     obj = {
#         'description': info['description'],
#         'relation': info['relation'],
#         'attributes': info['attributes'],
#         'data': data['data'],
#
#     }
#     arq1 = arff.dumps(obj)
#     arq = open(pasta + '/' + nome + '.arff', 'w')
#     arq.write(arq1)
#     arq.close()
#
#
# def cria_classificadores(X_train, y_train, repeticoes, dataset):#cria classificadores e os bags
#     #tree=DecisionTreeClassifier()
#     for x in range(repeticoes):
#         tree = DecisionTreeClassifier()
#         r = random.seed()
#         X_bag, X_yyy, y_bag, y_yyt = train_test_split(X_train, y_train, test_size=0.5, random_state=r, stratify=y_train)
#         dados = dict()
#         X = list()
#         for i in range(len(y_bag)):
#             linha_arff = list()
#             for j in X_bag[i]:
#                 linha_arff.append(j)
#             linha_arff.append(y_bag[i])
#             X.append(linha_arff)
#         dados['data'] = X
#         cria_arff(dataset, dados, "bags"+str(x), 'bag')
#         print('Criando bags-bag'+str(x))
#         tree.fit(X_bag,y_bag)
#         joblib.dump(tree, "clf/TreeClas"+str(x)+".pkl")#salva o classificador
#         print('Criando classificador-Tree' + str(x))
#
# def abre_arff(dataset):
#     labels = []
#     instancias = []
#     for i in range(len(dataset['data'])):  # percorre a base e separa os labels das classes
#         labels.append(dataset['data'][i][16])
#         labels = [str(w) for w in labels]  # transforma as classes em string
#     for i in dataset['data']:
#         instancias.append(i[:-1])  # salva so as instancias(sem classes)
#     return instancias, labels
#
# def carrega_classificadores():
#     global nome
#     nome=[]
#     for i in range(100):
#         nome.append('TreeClas'+str(i))
#         classifiers.append(joblib.load('clf/TreeClas'+str(i)+'.pkl'))
#     print ('carregando classificadores')
#
#     return classifiers, nome
#
# def cruzamento(ind1, ind2):
#
#     midsize = individual_size / 2
#     ind1 = ind1[0:midsize - 2] + ind2[midsize:individual_size - 1]
#     ind2 = ind2[0:midsize - 2] + ind1[midsize:individual_size - 1]
#     return creator.Individual(ind1), creator.Individual(ind2)
#
#
# def mutacao(individual):
#     idx_rand = random.randint(0, len(individual) - 1)
#     if (individual[idx_rand] == 1):
#         individual[idx_rand] = 0
#     else:
#         individual[idx_rand] = 1
#     return individual,
#
#
# def fitness(individual,type=''):
#     global current_ind
#     if type=='acu':
#         X=X_test
#         y=Y_test
#     else:
#         X=X_val
#         y=Y_val
#     correct = 0
#     total = 0
#     l1 = 1
#     maxl1 = len(X)
#     for i in range(len(X)):
#         pred_ensemble = list()
#         l2 = 1
#         maxl2 = len(individual)
#         for j in range(len(individual)):
#             # print("\rAvaliando individuo {}/{} instancia de validacao {}/{} classificador {}/{}".format(current_ind, 100, l1, maxl1, l2, maxl2), end="")
#             if (individual[j] == 1):
#                 pred_val = classifiers[j].predict_proba(np.array([X[i]]))
#                 pred_ensemble.append(pred_val)
#             l2 += 1
#         class_predicted = CombineBySum(pred_ensemble)
#         if (class_predicted == y[i]):
#             correct += 1
#         total += 1
#         l1 += 1
#     if (current_ind == 100):
#         current_ind = 0
#     else:
#         current_ind += 1
#     accuracy = float(correct) / total
#     return accuracy,
#
# def CombineBySum(results):
#     if (len(results) > 0):
#         if (len(results[0]) > 0):
#             vote_list = [0 for i in range(len(results[0]))]
#             for i in results:
#                 for j in range(len(i)):
#                     vote_list[j] += i[j]
#             return np.argmax(np.array(vote_list))
#         return -1
#     return -1
#
# ###############################################################################
#
# dataset = arff.load(open('letter.arff'))
# #X_train, X_test, X_val, Y_train, Y_test, Y_val =  cria_dataset(dataset)
# #cria_classificadores(X_train,Y_train,100,dataset)
#
# #carrega_classificadores()
# dataset1=arff.load(open('dataset/treino.arff'))
# dataset2=arff.load(open('dataset/teste.arff'))
# dataset3=arff.load(open("dataset/validacao.arff"))
# X_train,Y_train=abre_arff(dataset1)
# X_test,Y_test=abre_arff(dataset2)
# X_val,Y_val=abre_arff(dataset3)
#
# carrega_classificadores()
# individual_size = 10
# nr_generation = 10
# qt_selection = 2
# nr_children_generation = 30
# proba_crossover = 0.8
# proba_mutation = 0
#
# creator.create("Fitness", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.Fitness)
#
# toolbox = base.Toolbox()
#
# toolbox.register("attr_item", random.randint, 0, 1)
#
# toolbox.register("individual", tools.initRepeat, creator.Individual,
#                  toolbox.attr_item, individual_size)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
# toolbox.register("evaluate", fitness)
# toolbox.register("mate", cruzamento)
# toolbox.register("mutate", mutacao)
# toolbox.register("select", tools.selRoulette)
#
# pop = toolbox.population(n=qt_selection)
#
# hof = tools.ParetoFront()
# stats = tools.Statistics(lambda ind: ind.fitness.values)
#
# stats.register("avg", np.mean, axis=0)
# stats.register("std", np.std, axis=0)
# stats.register("min", np.min, axis=0)
# stats.register("max", np.max, axis=0)
# #
# algorithms.eaMuPlusLambda(pop, toolbox, qt_selection, nr_children_generation, proba_crossover, proba_mutation,
#                          nr_generation,stats,  halloffame=hof, verbose=True)
#
#
#
# #of
# print("Accuracy: {}".format(fitness(hof[0],type='acu')))
#
# #else:
#  #   maior, media, resultados, nome=acuracia(100,X_test,y_test)
# #print(random.randint(0, 1))
#
#   #  print(nome)
#   #  print (maior)
#    # print (resultados
#
#
# import random
#
# import numpy
#
# from deap import algorithms
# from deap import base
# from deap import creator
# from deap import tools
#
# IND_INIT_SIZE = 5
# MAX_ITEM = 50
# MAX_WEIGHT = 50
# NBR_ITEMS = 20
# items = {}
#
# X=dict()
# X['data']=list()
# X['data']=[[6.3487669999999996, 0.425784], [5.1449629999999997, 0.37743700000000002], [5.16805, 0.32056000000000001], [6.775487, 0.39499200000000001], [4.3326099999999999, 0.39253100000000002], [5.3018700000000001, 0.39150299999999999], [3.94746, 0.53991500000000003], [4.3883299999999998, 0.37757600000000002], [4.1567869999999996, 0.39092100000000002], [5.3685700000000001, 0.37168600000000002], [4.8684900000000004, 0.34349600000000002], [5.320417, 0.33280399999999999], [4.7127129999999999, 0.36907099999999998], [4.8196430000000001, 0.38705200000000001], [3.8781629999999998, 0.38196400000000003], [4.8193099999999998, 0.334565], [4.3666869999999998, 0.33706000000000003], [5.1203130000000003, 0.34611700000000001], [6.0060399999999996, 0.30915599999999999], [4.9825229999999996, 0.26452599999999998], [6.5112030000000001, 0.35540500000000003], [3.7537769999999999, 0.30435299999999998], [4.1090099999999996, 0.34690799999999999], [4.6495300000000004, 0.41454200000000002], [5.7606869999999999, 0.33205600000000002], [5.7038130000000002, 0.40967599999999998], [4.3395799999999998, 0.28986099999999998], [3.9159069999999998, 0.33878599999999998], [3.7715299999999998, 0.38927899999999999], [4.8476569999999999, 0.37155100000000002], [3.5167830000000002, 0.40375699999999998], [5.5378530000000001, 0.44293700000000003], [4.8996469999999999, 0.329982], [3.5970499999999999, 0.36419299999999999], [5.6934630000000004, 0.30298199999999997], [4.7788700000000004, 0.381851], [3.9963299999999999, 0.333708], [4.88293, 0.386683], [3.9880070000000001, 0.43211699999999997], [4.3001399999999999, 0.358319], [5.0011029999999996, 0.31623499999999999], [4.87073, 0.39901500000000001], [3.926253, 0.37820700000000002], [5.6741669999999997, 0.456459], [6.9707999999999997, 0.38848199999999999], [5.5051430000000003, 0.314716], [4.8891970000000002, 0.27029900000000001], [6.0373200000000002, 0.33265899999999998], [4.5573069999999998, 0.387735], [5.2916670000000003, 0.30687199999999998], [4.2710229999999996, 0.396233], [5.9998969999999998, 0.39691199999999999], [4.4874830000000001, 0.32508199999999998], [4.7250629999999996, 0.35096899999999998], [6.2637200000000002, 0.38924399999999998], [4.1027129999999996, 0.37238700000000002], [6.2873599999999996, 0.44101800000000002], [4.7395300000000002, 0.29914600000000002], [4.8010900000000003, 0.45089800000000002], [4.5587900000000001, 0.33936100000000002], [5.3658970000000004, 0.45315800000000001], [4.565493, 0.38689200000000001], [4.40862, 0.37664199999999998], [5.9509169999999996, 0.381996], [4.4668999999999999, 0.44090600000000002], [4.6956829999999998, 0.33458900000000003], [5.3151070000000002, 0.41484300000000002], [6.592123, 0.35420099999999999], [5.1035500000000003, 0.45615800000000001], [3.8610530000000001, 0.40954499999999999], [4.7449899999999996, 0.341804], [6.8692970000000004, 0.44997599999999999], [5.0852329999999997, 0.40162799999999999], [4.1507630000000004, 0.35521799999999998], [3.762807, 0.52426200000000001], [4.7335729999999998, 0.35832900000000001], [4.1132169999999997, 0.41217900000000002], [4.4727269999999999, 0.33685900000000002], [3.9264269999999999, 0.394177], [5.5614800000000004, 0.39984500000000001], [5.0490729999999999, 0.29800399999999999], [5.4955069999999999, 0.43969599999999998], [3.6676799999999998, 0.47488200000000003], [6.0340870000000004, 0.42160700000000001], [4.7304830000000004, 0.41639199999999998], [4.9035770000000003, 0.433425], [5.0800299999999998, 0.41171400000000002], [5.1188029999999998, 0.27497300000000002], [5.1127729999999998, 0.36963299999999999], [4.87277, 0.41467700000000002], [5.53329, 0.34553299999999998], [6.3614129999999998, 0.4521], [4.5839670000000003, 0.28956999999999999], [5.9976029999999998, 0.35764800000000002], [4.1097070000000002, 0.302757], [5.9725869999999999, 0.25062200000000001], [4.4031669999999998, 0.34944900000000001], [4.0628869999999999, 0.45813100000000001], [4.997007, 0.37108799999999997], [3.8264170000000002, 0.37974400000000003]]
# X['data'][0]=[1,1]
# for i in range(NBR_ITEMS):
#     items[i] = (random.randint(1, 10), random.uniform(0, 100))
#
# print(items)


# import random
#
# from deap import base
# from deap import creator
# from deap import tools
#
# IND_SIZE = 5
#
# creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
# creator.create("Individual", list, fitness=creator.FitnessMin)
#
# toolbox = base.Toolbox()
# toolbox.register("attr_float", random.randint,1,100)
# toolbox.register("individual", tools.initRepeat, creator.Individual,
#                  toolbox.attr_float, n=1)
# ind1 = toolbox.individual()
# print(ind1[0])






#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
"""This example shows a possible answer to a problem that can be found in this
xkcd comics: http://xkcd.com/287/. In the comic, the characters want to get
exactly 15.05$ worth of appetizers, as fast as possible."""
import random
from operator import attrgetter
from collections import Counter

# We delete the reduction function of the Counter because it doesn't copy added
# attributes. Because we create a class that inherit from the Counter, the
# fitness attribute was not copied by the deepcopy.
del Counter.__reduce__

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

IND_INIT_SIZE = 3

# Create the item dictionary: item id is an integer, and value is
# a (name, weight, value) 3-uple. Since the comic didn't specified a time for
# each menu item, random was called to generate a time.
ITEMS_NAME = "Mixed Fruit", "French Fries", "Side Salad", "Hot Wings", "Mozzarella Sticks", "Sampler Plate"
ITEMS_PRICE = 2.15, 2.75, 3.35, 3.55, 4.2, 5.8
ITEMS = dict((name, (price, random.uniform(1, 5))) for name, price in zip(ITEMS_NAME, ITEMS_PRICE))

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", Counter, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_item", random.choice, ITEMS_NAME)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalXKCD(individual, target_price):
    """Evaluates the fitness and return the error on the price and the time
    taken by the order if the chef can cook everything in parallel."""
    price = 0.0
    times = list()
    for item, number in individual.items():
        price += ITEMS[item][0] * number
        times.append(ITEMS[item][1])
    return abs(price - target_price), max(times)


def cxCounter(ind1, ind2, indpb):
    """Swaps the number of perticular items between two individuals"""
    for key in ITEMS.keys():
        if random.random() < indpb:
            ind1[key], ind2[key] = ind2[key], ind1[key]
    return ind1, ind2


def mutCounter(individual):
    """Adds or remove an item from an individual"""
    if random.random() > 0.5:
        individual.update([random.choice(ITEMS_NAME)])
    else:
        val = random.choice(ITEMS_NAME)
        individual.subtract([val])
        if individual[val] < 0:
            del individual[val]
    return individual,


toolbox.register("evaluate", evalXKCD, target_price=15.05)
toolbox.register("mate", cxCounter, indpb=0.5)
toolbox.register("mutate", mutCounter)
toolbox.register("select", tools.selNSGA2)


def main():
    NGEN = 40
    MU = 100
    LAMBDA = 200
    CXPB = 0.3
    MUTPB = 0.6

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()

    price_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    time_stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats = tools.MultiStatistics(price=price_stats, time=time_stats)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats, halloffame=hof)

    return pop, stats, hof


if __name__ == "__main__":
    _, _, hof = main()
    from matplotlib import pyplot as plt

    error_price = [i.fitness.values[0] for i in hof]
    time = [i.fitness.values[1] for i in hof]
    plt.plot(error_price, time, 'bo')
    plt.xlabel("Price difference")
    plt.ylabel("Total time")
    plt.show()



