import matplotlib.pyplot as plt
import numpy as np
import Cpx, novo_perceptron
from sklearn.ensemble import AdaBoostClassifier
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


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
        ax.contour(xx, yy, Z, colors="red", linewidths=0.3)
    else:
        ax.contourf(xx, yy, Z, **params)
    ax.set_xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    ax.set_ylim((np.min(X[:, 1]), np.max(X[:, 0])))


def plot_dataset(X, y, ax=None, title=None, **params):
    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25,
               edgecolor='k',**params)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if title is not None:
        ax.set_title(title)
    return ax


def Adaboost(X_train, y_train):

    ada=AdaBoostClassifier(novo_perceptron.PPerceptron(n_jobs=4, tol=1.0), n_estimators=100, algorithm='SAMME.R')
    ada.fit(X_train,y_train)
    poolAda=ada.estimators_

    return poolAda



local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
base_name="Lithuanian"

X,y,_,_=Cpx.open_data(base_name,local_dataset)

Treino=Cpx.open_training(local,base_name,1)

bags_ga="_distdiverlinear_teste_parada_dist"

bags=Cpx.open_bag(cpx_caminho+"3/",base_name+"")
bags2 = Cpx.open_bag(cpx_caminho+str(3)+"/", base_name + bags_ga)
test,val=Cpx.open_test_vali(local,base_name,"3")

X_t,y_t=Cpx.biuld_x_y(Treino,X,y)
Xval,y_val=Cpx.biuld_x_y(val,X,y)
Xtest,y_test=Cpx.biuld_x_y(test,X,y)

Xval=np.array(Xval)
y_val=np.array(y_val)

Xtest=np.array(Xtest)
y_test=np.array(y_test)

X=np.array(X)
y=np.array(y)

X_t=np.array(X_t)
y_t=np.array(y_t)

classifier=[]
classifier2=[]
classifier3=Adaboost(X_t,y_t)
for i in range(1,99):
    X_b, y_b = Cpx.biuld_x_y(bags['inst'][i], X, y)

    X_b=np.array(X_b)
    y_b=np.array(y_b)
    ca, sc = Cpx.biuld_classifier(X_b, y_b, Xval, y_val)
    classifier.append(ca)

for i in range(1,99):
    X_b2, y_b2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
    X_b2=np.array(X_b2)
    y_b2=np.array(y_b2)

    ca2, sc2 = Cpx.biuld_classifier(X_b2, y_b2, Xval, y_val)

    classifier2.append(ca2)


for clf in classifier:
    #print(sc)
    ax = plot_dataset(X_t, y_t, title="Lithuanian")


    plot_classifier_decision(ax, clf, Xval)

    #ax.set_xlim((0, 1))
   # ax.set_ylim((0, 1))

    #plt.savefig('Bag'+str(i)+'+.png', dpi=300)
    #ax.clear()
plt.show()
for clf in classifier2:
     ax2 = plot_dataset(X_t, y_t, title="Lithuanian")
     plot_classifier_decision(ax2, clf, Xval)
#     ax2.set_xlim((0, 1))
#     ax2.set_ylim((0, 1))
#
plt.show()

for clf in classifier3:
     ax3 = plot_dataset(X_t, y_t, title="Lithuanian")
     plot_classifier_decision(ax3, clf, Xval)
#     ax2.set_xlim((0, 1))
#     ax2.set_ylim((0, 1))
#
plt.show()
plt.tight_layout()

ax4 = plot_dataset(X_t, y_t, title="Lithuanian")

#     ax2.set_xlim((0, 1))
#     ax2.set_ylim((0, 1))
#
plt.show()