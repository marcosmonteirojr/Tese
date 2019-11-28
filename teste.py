# import numpy as np
# import Cpx
# file = open('DistdiverP2', "r")
#
# linhas=file.readlines()
#
# matrix2=[]
#
# for i in linhas:
#     print(i)
#     x=i.split("\t")
#     matrix = []
#     for j in x:
#         matrix.append(float(j))
#     matrix2.append(matrix)
#
# cont =1
# inicio=0
# fim=100
# dist=[]
# while cont != 21:
#     print((cont))
#     dist.append(Cpx.dispersion(matrix2[inicio:fim]))
#     cont=cont+1
#     inicio=fim
#     fim=fim+100
# resultado=[]
# for i in dist:
#     resultado.append(np.mean(i))
# print(resultado)
# print(len(resultado))
# exit(0)


from sklearn.ensemble import BaggingClassifier, VotingClassifier
import pandas
from sklearn.linear_model import Perceptron
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.tree import DecisionTreeClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = Perceptron()
num_trees = 100



model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

x=model.fit(X,Y)
a=[[1,2,3],[1,2,3]]
b=[0,1]


#print(x.estimators_[0])
#e=x.estimators_[0]
#print(e.predict(X))
#print(x.estimator_[0].predict(X))
#y=VotingClassifier(model.estimators_)
#y.score(X,Y)
y=EnsembleVoteClassifier(x.estimators_,refit=False)
y.fit(a,b)
y.predict(X)

print(y.score(X,Y))
