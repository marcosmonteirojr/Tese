import arff, numpy, random, os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
global resultados, media, maior
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.ensemble import VotingClassifier
classifiers = list()
resultados=[]
current_ind=1


# def acuracia(repeticao, X_test, y_test):#mede a acuracia do classificador, retorna a media, o maior, os resultados e o nome do maior
#     resultados = []
#     anterior=0
#     for i in range(repeticao):
#         classificador = joblib.load('clf/TreeClas' + str(i) + '.pkl')
#         atual=classificador.score(X_test,y_test)
#         resultados.append(atual)
#         if (atual>anterior):
#             anterior=atual
#             nome='TreeClas'+str(i)
#     maior=max(resultados)
#     media=numpy.mean(resultados)
#     return maior, media, resultados, nome

# def check_marcos(individual):
# 
#     c=[]
#     n=[]
# 
#     for j in range(len(individual)):
#         if (individual[j] == 1):
#             nome_val=nome[j]
#             pred_val = classifiers[j]
#             c.append(pred_val)
#             n.append(nome_val)
# 
#     eclf = VotingClassifier(estimators=zip(n,c),voting='hard')
#     c.append(eclf)
#     n.append('Ensemble')
#     scores=[]
#   #  print(zip(c,n))
#     for clf, label in zip(c,n):
#         scores.append( cross_val_score(clf, X_test, Y_test, cv=2, scoring='accuracy'))
#     return np.mean(scores),

# def evalEnsemble(individual):
#    # print (individual)
#     global current_ind
#     c=[]
#     n=[]
# 
#     for j in range(len(individual)):
#         if (individual[j] == 1):
#             nome_val=nome[j]
#             pred_val = classifiers[j]
#             c.append(pred_val)
#             n.append(nome_val)
# 
#     eclf = VotingClassifier(estimators=zip(n,c),voting='hard')
#     c.append(eclf)
#     n.append('Ensemble')
#     scores=[]
#   #  print(zip(c,n))
#     for clf, label in zip(c,n):
#         #scores.append( cross_val_score(clf, X_val, Y_val, cv=2, scoring='accuracy'))
# 
#     print("Accuracy: %f (+/- %0.2f) [%s]" % (np.mean(scores), np.std(scores), label))
#     #print(scores.mean())
# 
#     return np.mean(scores),

def cria_dataset(dataset): #cria treino teste e validacao e grava em arquivo
    print('Criando dataset\n')
    labels =[]
    instancias=[]
    for i in range(len(dataset['data'])):  # percorre a base e separa os labels das classes
        #labels.append(dataset['data'][i][16])
        #print (labels)
        #labels = [str(w) for w in labels]#transforma as classes em string
        labels.append(ord(dataset['data'][i][-1])-65)


    for i in dataset['data']:
        instancias.append(i[:-1])#salva so as instancias(sem classes)

    #tree = DecisionTreeClassifier()
    X_train, X_x, y_train, y_y = train_test_split(instancias, labels, test_size=0.5, random_state=True, stratify=labels)#divide a base de treino
    X_test, X_val, y_test, y_val= train_test_split(X_x, y_y, test_size=0.5, random_state=None, stratify=y_y)#divide a base entre teste e validacao
    # tree.fit(X_train,y_train)
    # print(tree.score(X_val,y_val))

    dados = dict()
    X=list()
    for i in range(len(y_train)):
        linha_arff=list()
        for j in X_train[i]:
            linha_arff.append(j)
        linha_arff.append(y_train[i])
        X.append(linha_arff)
    dados['data'] = X
    print ('Criando treino\n')
    cria_arff(dataset,dados,"treino",'dataset')


    dados = dict()
    X = list()
    for i in range(len(y_test)):
        linha_arff = list()
        for j in X_test[i]:
            linha_arff.append(j)
        linha_arff.append(y_test[i])
        X.append(linha_arff)
    dados['data'] = X
    print ('Criando teste\n')
    cria_arff(dataset, dados, "teste",'dataset')


    for i in range(len(y_val)):
        linha_arff = list()
        for j in X_val[i]:
            linha_arff.append(j)
        linha_arff.append(y_val[i])
        X.append(linha_arff)
    dados['data'] = X
    cria_arff(dataset, dados, "validacao",'dataset')
    print ('Criando validacao\n')
    return X_train, X_test, X_val, y_train, y_test, y_val # X_ -> bases sem labels, y_-> labels


def cria_arff(info, data, nome, pasta):
    obj = {
        'description': info['description'],
        'relation': info['relation'],
        'attributes': info['attributes'],
        'data': data['data'],

    }
    arq1 = arff.dumps(obj)
    arq = open(pasta + '/' + nome + '.arff', 'w')
    arq.write(arq1)
    arq.close()


def cria_classificadores(X_train, y_train, repeticoes, dataset):#cria classificadores e os bags
    #tree=DecisionTreeClassifier()
    for x in range(repeticoes):
        tree = DecisionTreeClassifier()
        r = random.seed()
        X_bag, X_yyy, y_bag, y_yyt = train_test_split(X_train, y_train, test_size=0.5, random_state=r, stratify=y_train)
        dados = dict()
        X = list()
        for i in range(len(y_bag)):
            linha_arff = list()
            for j in X_bag[i]:
                linha_arff.append(j)
            linha_arff.append(y_bag[i])
            X.append(linha_arff)
        dados['data'] = X
        cria_arff(dataset, dados, "bags"+str(x), 'bag')
        print('Criando bags-bag'+str(x))
        tree.fit(X_bag,y_bag)
        joblib.dump(tree, "clf/TreeClas"+str(x)+".pkl")#salva o classificador
        print('Criando classificador-Tree' + str(x))

def abre_arff(dataset):
    labels = []
    instancias = []
    for i in range(len(dataset['data'])):  # percorre a base e separa os labels das classes
        labels.append(dataset['data'][i][16])
        labels = [str(w) for w in labels]  # transforma as classes em string
    for i in dataset['data']:
        instancias.append(i[:-1])  # salva so as instancias(sem classes)
    return instancias, labels

def carrega_classificadores():
    global nome
    nome=[]
    for i in range(100):
        nome.append('TreeClas'+str(i))
        classifiers.append(joblib.load('clf/TreeClas'+str(i)+'.pkl'))
    print ('carregando classificadores')

    return classifiers, nome

def cruzamento(ind1, ind2):

    midsize = individual_size / 2
    ind1 = ind1[0:midsize - 2] + ind2[midsize:individual_size - 1]
    ind2 = ind2[0:midsize - 2] + ind1[midsize:individual_size - 1]
    return creator.Individual(ind1), creator.Individual(ind2)


def mutacao(individual):
    idx_rand = random.randint(0, len(individual) - 1)
    if (individual[idx_rand] == 1):
        individual[idx_rand] = 0
    else:
        individual[idx_rand] = 1
    return individual,


def fitness(individual,type=''):
    global current_ind
    if type=='acu':
        X=X_test
        y=Y_test
    else:
        X=X_val
        y=Y_val
    correct = 0
    total = 0
    l1 = 1
    maxl1 = len(X)
    for i in range(len(X)):
        pred_ensemble = list()
        l2 = 1
        maxl2 = len(individual)
        for j in range(len(individual)):
            # print("\rAvaliando individuo {}/{} instancia de validacao {}/{} classificador {}/{}".format(current_ind, 100, l1, maxl1, l2, maxl2), end="")
            if (individual[j] == 1):
                pred_val = classifiers[j].predict_proba(np.array([X[i]]))
                pred_ensemble.append(pred_val)
            l2 += 1
        class_predicted = CombineBySum(pred_ensemble)
        if (class_predicted == y[i]):
            correct += 1
        total += 1
        l1 += 1
    if (current_ind == 100):
        current_ind = 0
    else:
        current_ind += 1
    accuracy = float(correct) / total
    return accuracy,

def CombineBySum(results):
    if (len(results) > 0):
        if (len(results[0]) > 0):
            vote_list = [0 for i in range(len(results[0]))]
            for i in results:
                for j in range(len(i)):
                    vote_list[j] += i[j]
            return np.argmax(np.array(vote_list))
        return -1
    return -1

###############################################################################

dataset = arff.load(open('letter.arff'))
#X_train, X_test, X_val, Y_train, Y_test, Y_val =  cria_dataset(dataset)
#cria_classificadores(X_train,Y_train,100,dataset)

#carrega_classificadores()
dataset1=arff.load(open('dataset/treino.arff'))
dataset2=arff.load(open('dataset/teste.arff'))
dataset3=arff.load(open("dataset/validacao.arff"))
X_train,Y_train=abre_arff(dataset1)
X_test,Y_test=abre_arff(dataset2)
X_val,Y_val=abre_arff(dataset3)

carrega_classificadores()
individual_size = 10
nr_generation = 10
qt_selection = 2
nr_children_generation = 30
proba_crossover = 0.8
proba_mutation = 0

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

toolbox.register("attr_item", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, individual_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", cruzamento)
toolbox.register("mutate", mutacao)
toolbox.register("select", tools.selRoulette)

pop = toolbox.population(n=qt_selection)

hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
#
algorithms.eaMuPlusLambda(pop, toolbox, qt_selection, nr_children_generation, proba_crossover, proba_mutation,
                         nr_generation,stats,  halloffame=hof, verbose=True)



#of
print("Accuracy: {}".format(fitness(hof[0],type='acu')))

#else:
 #   maior, media, resultados, nome=acuracia(100,X_test,y_test)
#print(random.randint(0, 1))

  #  print(nome)
  #  print (maior)
   # print (resultados


