from sklearn.model_selection import train_test_split
from sklearn.linear_model import perceptron
#from sklearn.ensemble import bagging
import numpy as np
from deslib.util import diversity
import Marff, subprocess
import csv, random, os
from sklearn.utils import check_random_state
from scipy.spatial import distance

base_name="Haberman"
local_data="/media/marcos/Data/Tese/Bases2/Dataset/"


def open_data(base_name, local_data):

    dataset_file=Marff.abre_arff(local_data+base_name+".arff")
    X_data,y_data,dataset=Marff.retorna_instacias(dataset_file)

    dic=Marff.retorna_dic_data(dataset_file)
    del dic['data']
    return X_data,y_data,dataset,dic

def split_data(X_data,y_data):
    #
    indices = np.arange(len(X_data))
    X_train,X_temp,y_train,y_temp,id_train,id_temp=train_test_split(X_data,y_data, indices, test_size=0.5,stratify=y_data)
    X_test,X_vali,y_test,y_vali ,id_test,id_vali=train_test_split(X_temp,y_temp,id_temp,test_size=0.5, stratify=y_temp)
    del X_data,y_data,X_temp,y_temp, id_temp

    return X_train,y_train,X_test,y_test,X_vali,y_vali,id_train,id_test,id_vali

def biuld_bags(y_train, X_train=None, X_data=None, y_data=None,ind=None,types="ind"):
    #constroi os bags de forma randomica, duas formas por indices do traino, ou por instancias
    X = []
    y = []
    if types=="sample":

       # X_train=X_train.tolist()
       # y_train=y_train.tolist()

        max_samples=int(round((len(y_train)*0.5),1))
        random_state = check_random_state(random.seed())
        indices = random_state.randint(0, len(y_train), max_samples)

        for i in indices:
           X.append(X_train[i])
           y.append(y_train[i])
        #X=np.array(X)
        #y=np.array(y)
        #print(X)
        return X, y

    if types=="ind":
        #print(X_data)

        #X_data=X_data.tolist()
        #y_data=y_data.tolist()
        idx=random.choices(ind,k=int(round((len(y_train)*0.5),1)))
        for i in idx:
            X.append(X_data[i])
            y.append(y_data[i])

       # print(X)
        return X,y,idx
    #exit(0)
    #print(y)

def generate_csv(dic):
    #grava um arquivo csv com o nome dos atributos na primeira linha, e das instancias, arquivo para script do R
    with open("/media/marcos/Data/Tese/teste.csv",'w') as f:
        w = csv.writer(f)
        w.writerow(dic['class'])
        w.writerows(dic['data'])

def complexity_data():

    proc = subprocess.Popen(['Rscript /home/marcos/Documentos/new_3.r'],
                            stdout=subprocess.PIPE, shell=True)
    (cont_arq, err) = proc.communicate()
    cont_arq = (cont_arq.decode("utf-8"))
    print('cpx')
    cont_arq = cont_arq.split()
    complex = []
   # x=[]
    for i in cont_arq:
        if (i[:3] == "ove" or i[:3] == "nei" or i[:3] == "dim" or i[:3] == "lin" or i[:3] == "bal" or i[:3] == "net"):
            #x.append(i)
            continue
        else:
            #print(i)
            complex.append(float(i))
   # print(x)
    #exit(0)
    return complex

def biuld_dic(X,y, dic):
    #constroi o dicionario para o construir o csv do R (instancias e classes)
    d = dict()
    g = []
    g.append(y)
    g = np.array(g)
    g.astype(int)
    # X=np.array(X)
   # print(g)
    d['class'] = dic['class']
    d['data'] = np.concatenate((X, g.T), axis=1)
    d['data'] = d['data'].tolist()

    return d

def biuld_classifier(X_train, y_train, X_val, y_val):
    #constroi os classificadores, e retorna classificador, score e predict
    perc = perceptron.Perceptron(n_jobs=4, max_iter=100)
    perc.fit(X_train, y_train)
    score=perc.score(X_val,y_val)
    predict=perc.predict(X_val)

    return perc, score, predict

def dispersion(complexity):

    result=[]
    for i in range(len(complexity)):
        dist = []
        for j in range(len(complexity)):
            if i==j:
                continue
            else:
                dist.append(distance.euclidean(complexity[i],complexity[j]))
        result.append(np.mean(dist))

    return result

def diversitys(y_test,predicts):

    q_test=[]
    double_faults=[]
    for i in range(len(predicts)):
        q=[]
        db=[]
        for j in range(len(predicts)):
            if i==j:
                continue
            else:
               q.append(diversity.Q_statistic(y_test,predicts[i],predicts[j]))
               db.append(diversity.double_fault(y_test,predicts[i],predicts[j]))
        q_test.append(np.mean(q))
        double_faults.append(np.mean(db))

    return q_test, double_faults

def biuld_csv_result(complexity_result,score,Q_test, Df,disp):
    global base_name
    header=['overlapping.F1', 'overlapping.F1v', 'overlapping.F2', 'overlapping.F3', 'overlapping.F4', 'neighborhood.N1', 'neighborhood.N2', 'neighborhood.N3', 'neighborhood.N4', 'neighborhood.T1', 'neighborhood.LSCAvg', 'linearity.L1', 'linearity.L2', 'linearity.L3', 'dimensionality.T2', 'dimensionality.T3', 'dimensionality.T4', 'balance.C1', 'balance.C2', 'network.Density', 'network.ClsCoef', 'network.Hubs','Score', 'Q_test','DoubleFault','Disper']
    for i in range(len(complexity_result)):
        complexity_result[i].append(score[i])
        complexity_result[i].append(Q_test[i])
        complexity_result[i].append(Df[i])
        complexity_result[i].append(disp[i])
    complexity_result=np.round(complexity_result,3)
    with open("/media/marcos/Data/Tese/Resultados_"+base_name+".csv",'a') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(complexity_result)

def save_bag(inds,types,local,base_name, iteration):

    if types=='validation':
        if (os.path.exists(local+"/Validacao/"+str(iteration)) == False):
            os.system("mkdir -p " + local+"/"+str(iteration))
            with open(local+"/" +str(iteration)+"/"+ base_name + ".csv", 'w') as f:
                w = csv.writer(f)
                w.writerow(inds)

    if types=="test":
        if (os.path.exists(local + "/Teste/"+str(iteration)) == False):
            os.system("mkdir -p " + local+"/"+str(iteration))
            with open(local+"/"+str(iteration)+"/" + base_name + ".csv", 'w') as f:
                w = csv.writer(f)
                w.writerow(inds)

    if types=="train":
        if (os.path.exists(local + "/Treino/"+str(iteration)) == False):
            os.system("mkdir -p " + local+"/"+str(iteration))
            with open(local+"/" +str(iteration)+"/"+ base_name + ".csv", 'w') as f:
                w = csv.writer(f)
                w.writerow(inds)

    if types == "bags":
        if (os.path.exists(local + "/Bags/"+str(iteration)) == False):
            os.system("mkdir -p " + local+"/"+str(iteration))
            with open(local + "/" +str(iteration)+"/"+ base_name + ".csv", 'a') as f:
                w = csv.writer(f)
                w.writerow(inds)

def oracle(predict,y_val):
    a=[]
    result=[]

    for i in range(len(predict)):
        for j in range(predict[i]):
            if predict[i][j]==y_val[j]:
                print(predict[i][j],y_val)
                a.append(1)
            else:
                a.append(0)
        result.append(a)
    return result

def routine_save_bags(local_dataset, local ,base_name, iteration ):
    #rotina para criar treino teste e validaçao alem dos 100 bags, local e onde esta o dataset orig
    X_data, y_data, dataset, dic = open_data(base_name, local_dataset)
    X_train, y_train, X_test, y_test, X_vali, y_vali, id_train, id_test, id_vali = split_data(X_data, y_data)

    save_bag(id_train, 'train', local+"/Treino/",base_name,(iteration))
    save_bag(id_vali, 'validation', local+"/Validacao/",base_name,str(iteration))
    save_bag(id_test, 'test', local+"/Teste/",base_name,str(iteration))
    for i in range(1, 101):
        X_bag, y_bag, id = biuld_bags(y_train, X_data=X_data, y_data=y_data, ind=id_train, types="ind")
        id.insert(0,i)
        #print(id)
        save_bag(id, 'bags', local+"/Bags/", base_name, str(iteration))
    return  X_train, y_train, X_test, y_test, X_vali, y_vali, dic

def open_bag(local_bag, base_name):
    bags = dict()
    bags['nome'] = list()
    bags['inst'] = list()
    with open(local_bag+base_name+'.csv', 'r') as f:
        reader = csv.reader(f)
        indx = list(reader)
    for i in indx:
        bags['nome'].append(i[0])
        bags['inst'].append(i[1:])
    #print(indx[0])
   # print(bags['inst'][0])
    return bags

def main():
    import time
    inicio = time.time()
    X_data, y_data, dataset, dic = open_data(base_name, local_data)
    X_train, y_train, X_test, y_test, X_vali, y_vali, id_train, id_test, id_vali = split_data(X_data, y_data)
    complexity_result=[]
    score=[]
    predict=[]
   # exit(0)

   # print(X_test)

   # for i in range(1,10):
    #print(i)
    X_bag,y_bag=biuld_bags(y_train=y_train,X_train=X_train,types="sample")
        #print((X_bag))
        #exit(0)
    newdic=biuld_dic(X_bag,y_bag,dic)
    generate_csv(newdic)
        #save_bag(id_train,'train',"/media/marcos/Data/Tese/Bases3/Treino",base_name)
    complexity_result.append(complexity_data())
       # _,sc,pre=biuld_classifier(X_bag,y_bag,X_vali,y_vali)
       # score.append(sc)
       ##oracle(pre,y_vali)

    #q,df=diversitys(y_test,predict)
    #disp = dispersion(complexity_result)
    #biuld_csv_result(complexity_result,score,q,df,disp)
    complexity_result = []
    score = []
    predict = []
    fim = time.time()
    print((fim - inicio)/60)
    #print(complexity_result)
    #print((disp))
    #print(len(q))




if __name__ == "__main__":
    main()


# dic=(biuld_dic(X_vali,y_vali, dic))
# generate_csv(dic)
# print(complexity_data())
#
# perc, score, predict= biuld_classifier(X_train,y_train,X_test,y_test)
# perc1, score1, predict1= biuld_classifier(X_train,y_train,X_test,y_test)
#
#
# #print(perc)
# print(score)
# print(predict)
# #print(perc1)
# print(score1)
# print(predict1)
# print(diversity.double_fault(y_test,predict,predict1))
# print(diversity.Q_statistic(y_test,predict,predict1))
# #print(y_data)

#dic['data']=np.reshape(dic['data'],(len(dic['data']),len(dic['data'][0])))
#print(dic['data'])

#dfx = pd.DataFrame(X_data)
#y_targ = robjects.IntVector(y_data)
#y_dat=str(y_data)
#y_data=str(y_data).strip('[]')
#y_data= ''.join(map(str, y_data))
#print(y_data)
#exit(0)
#y_data=(str(dic['class']).strip('[]'))
#dfx.columns=dic['class']
#print(dfx)

#y_data=str(dic['class'])

#X_data=pd.DataFrame.to_string(dfx,index=False)
#print(X_data)
#X_data=str(dic['data'])
#print(dic['class'])


#command='Rscript'
##path='/home/marcos/Documentos/new_3.r'
#args=["/media/marcos/Data/Tese/teste.csv"]
#cmd=[command, path] + args
#print(cmd)
#my_env = os.environ.copy()
#my_env['R_HOME'] = '/home/marcos/anaconda3/envs/tese2/lib/R' + my_env['R_HOME']
#x=subprocess.check_output(cmd,universal_newlines=True,env=my_env)
