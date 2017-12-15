#!/usr/bin/python
#
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
#
from __future__ import print_function
import random
import arff
import sys
import numpy as np
import time
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
#
#
def print_parameters(clf):
    print("Best parameters with score {:.5f}% set found on development set:".format(clf.best_score_))
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#
def LoadLetters(arff_file):
    f = open(arff_file, "r")
    letters = arff.load(f)
    X = list()
    Y = list()
    for i in letters['data']:
        x_temp = list()
        for j in range(len(i)-1):
            x_temp.append(i[j])
        Y.append(ord(i[-1])-65)
        X.append(x_temp)
    f.close()
    return X,Y
#
def Classifier(X,Y):
#    tuned_parameters = [{'splitter': ['best'], 'criterion': ['entropy'], 'max_depth': [10,30,60], 'max_leaf_nodes': [30,90,150, 250, 500, 750, 1000], 'max_features': [0.5, 0.7]}]
#    tuned_parameters = [{'splitter': ['random'], 'max_depth': [3,6,9],
#                    'max_leaf_nodes': [3,6,9], 'max_features': ['sqrt','log2'],
#                    'criterion': ['gini']}]
#    tuned_parameters = [{'splitter': ['random'], 'max_depth': [6,9],
#                    'max_leaf_nodes': [6,9], 'max_features': ['auto'],
#                    'criterion': ['gini']}]
#    c45 = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=10)
    c45 = tree.DecisionTreeClassifier()
    c45.fit(X,Y)
    return c45
#
def SplitDataset(X,Y):
    X_train_val,X_test,Y_train_val,Y_test = train_test_split(X,Y,test_size=0.25)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.33)
    del X_train_val
    del Y_train_val
    return X_train, Y_train, X_test, Y_test, X_val, Y_val
#
def CreateBags(X_train, Y_train, nr_bags=100, bag_size=0.5):
    idx_lists = list()
    for i in range(nr_bags):
        if(walking_verbose == True):
            print("\rGerando indices do bag: {}/{}".format(i+1, nr_bags), end="")
        idx_list = list()
        while(len(idx_list) < (len(X_train)*bag_size)):
            x = random.randint(0,len(X_train)-1)
	    if(x not in idx_list or bags_with_replacement==True):
            	idx_list.append(x)
        idx_lists.append(idx_list)
    print()
    X_train_bags = list()
    Y_train_bags = list()
    l1 = 1
    maxl1 = len(idx_lists)
    for i in idx_lists:
        X_trains = list()
        Y_trains = list()
        l2 = 1
        maxl2 = len(i)
        for j in i:
            if(walking_verbose == True):
                print("\rGerando bag {}/{} instancia {}/{}".format(l1,maxl1,l2,maxl2), end="")
            X_trains.append(X_train[j])
            Y_trains.append(Y_train[j])
            l2 += 1
        X_train_bags.append(X_trains)
        Y_train_bags.append(Y_trains)
        l1 += 1
    print()
    return X_train_bags,Y_train_bags
#
def SaveBags(X_bags_train, Y_bags_train, directory):
    for i in range(len(X_bags_train)):
        SaveSet(X_bags_train[i], Y_bags_train[i], "train_{}.txt".format(i), directory)
#
def SaveSet(X, Y, file_name, directory):
    if( os.path.exists(directory) == False ):
        try:
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise
    f = open(directory+"/"+file_name, "w")
    for i in range(len(Y)):
        f.write("{}".format(Y[i]))
        for j in X[i]:
            f.write(";{}".format(j))
        f.write("\n")
    f.close()
#
def SaveClassifiers(classifiers, directory):
    if( os.path.exists(directory) == False ):
        try:
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise
    joblib.dump(classifiers, directory+"/classifiers.pkl")
    return 1
#
def LoadClassifiers(directory):
    classifiers = joblib.load(directory+"/classifiers.pkl")
    return classifiers
#
# Combine results by vote
#
def CombineByVote(results):
    if(len(results)>0):
        if(len(results[0])>0):
            vote_list = [0 for i in range(len(results[0]))]
            for i in results:
                vote_list[np.argmax(np.array(i))] += 1
            return np.argmax(np.array(vote_list))
        return -1
    return -1
#
# Combine results by sum
#
def CombineBySum(results):
    if(len(results)>0):
        if(len(results[0])>0):
            vote_list = [0 for i in range(len(results[0]))]
            for i in results:
                for j in range(len(i)):
                    vote_list[j] += i[j]
            return np.argmax(np.array(vote_list))
        return -1
    return -1
#
# Combine results by product
#
def CombineByProduct(results):
    vote_list = [0 for i in range(len(results))]
    for i in range(len(results)):
        for j in len(range(results[i])):
            vote_list[j] *= results[i][j]
    return np.argmax(np.array(vote_list))
#
# Combine by max
#
def CombineByMax(results):
    return 0
#
def evalEnsemble(individual):
    global current_ind
    correct = 0
    total = 0
    l1 = 1
    maxl1 = len(X_val)
    for i in range(len(X_val)):
        pred_ensemble = list()
        l2 = 1
        maxl2 = len(individual)
        for j in range(len(individual)):
            if(walking_verbose == True):
                print("\rAvaliando individuo {}/{} instancia de validacao {}/{} classificador {}/{}".format(current_ind, individual_size, l1, maxl1, l2, maxl2), end="")
            if(individual[j] == 1):
                pred_val = classifiers[j].predict_proba(np.array([X_val[i]]))
                pred_ensemble.append(np.squeeze(pred_val))
            l2 += 1
        class_predicted = CombineByVote(pred_ensemble)
        if(class_predicted == Y_val[i]):
            correct += 1
        total += 1
        l1 += 1
    if(current_ind == individual_size):
        current_ind = 0
    else:
        current_ind +=1
    accuracy = float(correct)/total
    return accuracy,
#
def evalEnsembleFast(individual):
    global current_ind
    correct = 0
    total = 0
    l1 = 1
    maxl1 = len(X_val)
    for i in range(len(X_val)):
        pred_ensemble = list()
        l2 = 1
        maxl2 = len(individual)
        for j in range(len(individual)):
            if(walking_verbose == True):
                print("\rAvaliando individuo {}/{} instancia de validacao {}/{} classificador {}/{}".format(current_ind, individual_size, l1, maxl1, l2, maxl2), end="")
            if(individual[j] == 1):
                pred_val = all_preds[j][i]
                pred_ensemble.append(np.squeeze(pred_val))
            l2 += 1
        class_predicted = CombineByVote(pred_ensemble)
        if(class_predicted == Y_val[i]):
            correct += 1
        total += 1
        l1 += 1
    if(current_ind == individual_size):
        current_ind = 0
    else:
        current_ind +=1
    accuracy = float(correct)/total
    nr_classifiers = 0
    for i in individual:
        if(i == 1):
            nr_classifiers += 1
    fill_rate = float(nr_classifiers)/individual_size
    return accuracy,fill_rate
#
def cxEnsemble(ind1, ind2):
    midsize = individual_size/2
    ind1 = ind1[0:midsize-2]+ind2[midsize:individual_size-1]
    ind2 = ind2[0:midsize-2]+ind1[midsize:individual_size-1]
    return creator.Individual(ind1), creator.Individual(ind2)
#
def mutEnsemble(individual):
    idx_rand = random.randint(0,len(individual)-1)
    if(individual[idx_rand] == 1):
        individual[idx_rand] = 0
    else:
        individual[idx_rand] = 1
    return individual,
#
def LoadSavedDatasets(directory):
    f = open(directory+"/test.txt", "r")
    Y_test = list()
    X_test = list()
    for i in f:
        Y_test.append(int(i.split(";")[0]))
        x_temp = list()
        for j in i.split(";")[1:]:
            x_temp.append(int(j))
        X_test.append(x_temp)
    f.close()
    #
    f = open(directory+"/val.txt", "r")
    Y_val = list()
    X_val = list()
    for i in f:
        Y_val.append(int(i.split(";")[0]))
        x_temp = list()
        for j in i.split(";")[1:]:
            x_temp.append(int(j))
        X_val.append(x_temp)
    f.close()
    #
    X_train_bag = list()
    Y_train_bag = list()
    files = os.listdir(directory)
    train_files = list()
    for i in files:
        if(str(i).find("train") != -1):
            train_files.append(i)
    for i in range(len(train_files)):
        f = open(directory+"/train_"+str(i)+".txt","r")
        Y_test = list()
        X_test = list()
        for i in f:
            Y_test.append(int(i.split(";")[0]))
            x_temp = list()
            for j in i.split(";")[1:]:
                x_temp.append(int(j))
            X_test.append(x_temp)
        f.close()
        X_train_bag.append(X_test)
        Y_train_bag.append(Y_test)
    print()
    return X_train_bag, Y_train_bag, X_test, Y_test, X_val, Y_val
#
def CheckAccuracy(X_test, Y_test, individual):
    global current_ind
    total = 0
    correct = 0
    l1 = 1
    maxl1 = len(X_test)
    for i in range(len(X_test)):
        pred_ensemble = list()
        l2 = 1
        maxl2 = len(individual)
        for j in range(len(individual)):
            if(walking_verbose == True):
                print("\rAvaliando individuo {}/{} instancia de teste {}/{} classificador {}/{}".format(current_ind, individual_size, l1, maxl1, l2, maxl2), end="")
            if(individual[j] == 1):
                pred_test = classifiers[j].predict_proba(np.array([X_test[i]]))
                pred_ensemble.append(np.squeeze(pred_test))
            l2 += 1
        class_predicted = CombineByVote(pred_ensemble)
        if(class_predicted == Y_test[i]):
            correct += 1
        total += 1
        l1 += 1
    if(current_ind == individual_size):
        current_ind = 0
    else:
        current_ind +=1
    accuracy = float(correct)/total
    print()
    return accuracy
#
def ClassifyAllInstances(clfs, X_val, Y_val):
    pred_clfs = list()
    for i in range(len(clfs)):
	if(walking_verbose == True):
	    print("\rPredicting for classifiers {}/{}".format(i,len(clfs)), end="")
	preds = list()
	for j in range(len(X_val)):
	    preds.append(np.squeeze(clfs[i].predict_proba(np.array([X_val[j]]))))
	pred_clfs.append(preds)
    print()
    return pred_clfs
#
# ########################################################################
# MAIN
# ########################################################################
#
if(len(sys.argv) != 2):
    print("smc_ga.py [arf_letters_file or directory_with_bags_and_classifiers]")
    exit(0)
#
hourtime = time.strftime("Iniciando algoritmo de bags: %H:%M:%S %d/%m/%Y")
print(hourtime)
print("Argumentos: {}".format(sys.argv))
sys.stdout.flush()
#
random.seed(64)
verbose = True
walking_verbose = False
individual_size = 100
bags_with_replacement = False
nr_generation = 50
qt_selection = 6 # (elitismo)
nr_children_generation = 100
proba_crossover = 0.8
proba_mutation = 0.01
current_ind = 1
#
load = 0
if( os.path.exists(sys.argv[1]) == True ):
    if os.path.isdir(sys.argv[1]):
        load = 1
#
if(load == 1):
    X_train_bags, Y_train_bags, X_test, Y_test, X_val, Y_val = LoadSavedDatasets(sys.argv[1])
    classifiers = joblib.load(sys.argv[1]+"/classifiers.pkl")
    #all_preds = ClassifyAllInstances(classifiers, X_val, Y_val)
else:
    #
    if(verbose == True):
        print("Carregando base")
    X, Y = LoadLetters(sys.argv[1])
    #
    if(verbose == True):
        print("Separando base em treino/teste/validacao")
    X_train, Y_train, X_test, Y_test, X_val, Y_val = SplitDataset(X,Y)
    #
    if(verbose == True):
        print("Criando bags")
    X_train_bags, Y_train_bags = CreateBags(X_train, Y_train, nr_bags=individual_size, bag_size=0.6)
    #
    directory = time.strftime("bags_%Y%m%d_%H%M%S")
    #
    if(verbose == True):
        print("Salvando arquivos de dados")
    SaveBags(X_train_bags, Y_train_bags, directory)
    #
    SaveSet(X_test, Y_test, "test.txt", directory)
    #
    SaveSet(X_val, Y_val, "val.txt", directory)
    #
    print()
    if(verbose == True):
        print("Criando classificadores")
    classifiers = list()
    imax = len(X_train_bags)
    for i in range(len(X_train_bags)):
        if(walking_verbose == True):
            print("\rTreinando classificador {}/{}".format(i+1,imax), end="")
        classifiers.append(Classifier(X_train_bags[i], Y_train_bags[i]))
    print()
    if(verbose == True):
        print("Salvando classificadores")
    sys.stdout.flush()

    joblib.dump(classifiers, directory+"/classifiers.pkl")

    print()
    correct = 0
    total = 0
#    tuned_parameters = [{'splitter': ['best'],
#                    'max_leaf_nodes': [9], 'max_features': ['auto'],
#                    'criterion': ['gini']}]
#    tuned_parameters = [{'splitter': ['random'], 'max_depth': [6,9],
#                    'max_leaf_nodes': [6,9], 'max_features': ['auto'],
#                    'criterion': ['gini']}]
#    c45 = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
    c45 = tree.DecisionTreeClassifier()
    c45.fit(X_train, Y_train)
    for i in range(len(X_test)):
        pred = c45.predict_proba(np.array([X_test[i]]))
        if(np.argmax(pred) == Y_test[i]):
            correct +=1
        total += 1
    print("Accuracia para CART simples: {:.4f}".format(float(correct)/total))
    sys.stdout.flush()
    #exit(0)
#
# Predicao antecipada para melhorar o desempenho
#
all_preds = ClassifyAllInstances(classifiers, X_val, Y_val)
#
hourtime = time.strftime("Iniciando algoritmo de GA: %H:%M:%S %d/%m/%Y")
print(hourtime)
#
#
#
#random.seed(1)
hourtime = time.strftime("Iniciando algoritmo de GA: %H:%M:%S %d/%m/%Y")
print(hourtime)

creator.create("Fitness", base.Fitness, weights=(1.0,-0.7))
creator.create("Individual", list, fitness=creator.Fitness)
#creator.create("Individual", int, fitness=creator.Fitness)
toolbox = base.Toolbox()

toolbox.register("attr_item", random.randint, 0, 1)# virar 123....


toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_item, individual_size)

#toolbox.register("individual", tools.initRepeat, creator.Individual,
#    toolbox.attr_item, 1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalEnsembleFast)
toolbox.register("mate", cxEnsemble)
toolbox.register("mutate", mutEnsemble)
toolbox.register("select", tools.selNSGA2)
#toolbox.register("select", tools.selRoulette)


pop = toolbox.population(n=qt_selection)

hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

if(verbose == True):
    print("Iniciando GA")
    sys.stdout.flush()

algorithms.eaMuPlusLambda(pop, toolbox, qt_selection, nr_children_generation, proba_crossover, proba_mutation, nr_generation, stats,
                          halloffame=hof, verbose=True)
#
hourtime = time.strftime("Finalizado algoritmo de GA: %H:%M:%S %d/%m/%Y")
print(hourtime)
sys.stdout.flush()
#
#
#
for i in range(len(hof)):
    print("Accuracy {}: {} {}".format(i, CheckAccuracy(X_test, Y_test, hof[i]), hof[i]))
    sys.stdout.flush()
#
classif_list = [1 for i in range(individual_size)]
print("Accuracy all: {} {}".format(CheckAccuracy(X_test, Y_test, classif_list), classif_list))


