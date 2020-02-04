from sklearn.ensemble import  BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import Cpx, numpy as np
from deslib.dcs import LCA
import random
#base=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_sample.csv","r")
#lable=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_labels2.csv","r")

base=open("/media/marcos/Data/Tese/Bases4/Dataset/features_tcnn_breakhis/features_dsfold1-40-train.txt","r")
lable=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_labels.csv","r")

#base=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_sample.csv","r")
#lable=open("/media/marcos/Data/Tese/Bases4/Dataset/Jaffe_labels.csv","r")
#random.seed(64)
y=lable.readline()

y=y[:-1].split(",")
y=[float(elem) for elem in y]
y=[int(elem) for elem in y]
sample=[]
#print(y)
for i in base:
    x=i[:-1].split(";")
    x=[float(elem) for elem in x]
   # print(x)
    sample.append(x)
print(sample)
#exit(0)
sample=np.array(sample)
y=np.array(y)
#print(y)
scores=[]
#for i in range (20):
##    print(i)
#X_train, y_train, X_test, y_test, X_vali, y_vali, *_ = Cpx.split_data(sample,y)
#print(y_vali)
#print(y_train)
#clf_svm = BaggingClassifier(base_estimator=OneVsRestClassifier(svm.SVC(C=1e-6, kernel="linear"),n_jobs=6),n_estimators=100, random_state=0, max_samples=0.5)
#clf_svm.fit(X_train, y_train)
#pred=clf_svm.predict(X_test,y_test)

#    scores.append(clf_svm.score(X_test,y_test))
#    print(scores)
#
#print(np.average(scores))
#clf_perc = BaggingClassifier(base_estimator=Perceptron(n_jobs=4,tol=1.0),n_estimators=100, random_state=0, max_samples=0.5)
#clf_perc.fit(X_train, y_train)



#poolBag=clf_perc.estimators_


#print(poolBag)

#print(clf_svm.score(X_test,y_test))


#lcab=LCA(poolBag)
#lcab.fit(X_vali, y_vali)
#print(lcab.score(X_test,y_test))