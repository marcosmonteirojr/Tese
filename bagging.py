from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.util.diversity import double_fault
from sklearn.calibration import CalibratedClassifierCV
import Marff

nome_base='Wine'
def abre_arff(nome_base):
    caminho='/media/marcos/Data/Tese/Bases2/Dataset/'+nome_base+".arff"
    dataset=Marff.abre_arff(caminho)
    X_data,y_data=Marff.retorna_instacias(dataset,np_array=True)
    #print(y_data)
    #exit(0)
    return X_data,y_data

def SplitDataset(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=64)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=64)
    del X_temp
    del y_temp
    print(y_train)
    #exit(0)
    return X_train, y_train, X_test, y_test, X_val,y_val

def new_bags(X_train, y_train,X_val):
    model = CalibratedClassifierCV(Perceptron(max_iter=10))
    bagging=BaggingClassifier(model,n_estimators=100,max_samples=.5,random_state=64)

    estimators=bagging.fit(X_train,y_train)

   # print(bagging.predict(X_train))
    print(bagging.estimators_[99].predict(X_test))

    return estimators
    #print(bagging.estimators_)

X,y=abre_arff(nome_base)

#print(len(y))

X_train,y_train,X_test,y_test,X_val,y_val=SplitDataset(X,y)

#print(len(y_train))

pool_classifiers=new_bags(X_train,y_train, X_test)



#print('Classification accuracy KNORA-Union: ', knorau.score(X_test, y_test))
#print('Classification accuracy KNORA-Eliminate: ', kne.score(X_test, y_test))



