import matplotlib.pyplot as plt
import numpy as np
import Cpx
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Importing DS techniques
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.static import StackedClassifier
from deslib.util.datasets import make_P2


# Plotting-related functions
def make_grid(x, y, h=.01):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_classifier_decision(ax, clf, X, mode='line', **params):
    xx, yy = make_grid(X[:, 0], X[:, 1])

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if mode == 'line':
        ax.contour(xx, yy, Z,linewidths=0.7,colors="red")
    else:
        ax.contourf(xx, yy, Z, **params)
    ax.set_xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    ax.set_ylim((np.min(X[:, 1]), np.max(X[:, 0])))


def plot_dataset(X, y, ax=None, title=None, **params):
    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25,
               edgecolor='k', **params)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if title is not None:
        ax.set_title(title)
    return ax

local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
base_name="Banana"

X,y,_,_=Cpx.open_data(base_name,local_dataset)

Treino=Cpx.open_training(local,base_name,1)
#X_t,y_t=Cpx.biuld_x_y(Treino,X,y)
bags=Cpx.open_bag(cpx_caminho+"1/",base_name+"")
val,test=Cpx.open_test_vali(local,base_name,"1")

Xval,y_val=Cpx.biuld_x_y(test,X,y)
Xval=np.array(Xval)
y_val=np.array(y_val)




classifier=[]
print(len(bags['inst']))
for i in range(10):
    print(i)
    X_t, y_t = Cpx.biuld_x_y(bags['inst'][i], X, y)
    X_t=np.array(X_t)
    y_t=np.array(y_t)
    ax = plot_dataset(Xval, y_val, title='Banana')
    ca,sc,_=Cpx.biuld_classifier(X_t,y_t,Xval,y_val)
    #    classifier.append(ca)

    #for clf in classifier:
    print(sc)
    plot_classifier_decision(ax, ca, Xval)
    #ax.set_xlim((0, 1))
   # ax.set_ylim((0, 1))

    plt.show()
    plt.tight_layout()