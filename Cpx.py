from sklearn.model_selection import train_test_split
from sklearn.linear_model import perceptron
#from sklearn.ensemble import bagging
from deslib.static.oracle import Oracle
import numpy as np
from deslib.util import diversity
import Marff, subprocess
import csv, random, os
from sklearn.utils import check_random_state
from multiprocessing import Pool
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise
os.environ['R_HOME'] = '/home/marcos/anaconda3/envs/tese2/lib/R'
import pandas as pd
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects.packages as rpackages
ecol = rpackages.importr('ECoL')
import rpy2.robjects as robjects


#base_name="Haberman"
#local_data="/media/marcos/Data/Tese/Bases2/Dataset/"


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

def biuld_bags_stratify(y_train, X_train=None, X_data=None, y_data=None,ind=None,types="ind"):
    print('atencao pq nao e a funcao de bag oficial')
    X = []
    y = []
    if types == "sample":

        max_samples = int(round((len(y_train) * 0.5), 1))
        random_state = check_random_state(random.seed())
        indices = random_state.randint(0, len(y_train), max_samples)

        for i in indices:
            X.append(X_train[i])
            y.append(y_train[i])
        # print(y)

        return X, y

    if types == "ind":
        indices=ind
        X_bag,xx,y_bag,yy,idx,idxx=train_test_split(X_train, y_train, indices, test_size=0.5, stratify=y_train)
       # print(indices)
       # print(idx)
        idx=idx.tolist()
        for i in idx:
            X.append(X_data[i])
            y.append(y_data[i])
        #exit(0)
        return X, y, idx

def biuld_bags(y_train, X_train=None, X_data=None, y_data=None,ind=None,types="ind"):
    #constroi os bags de forma randomica, duas formas por indices do treino, ou por instancias
    X = []
    y = []
    if types=="sample":

        max_samples=int(round((len(y_train)*0.5),1))
        random_state = check_random_state(random.seed())
        indices = random_state.randint(0, len(y_train), max_samples)

        for i in indices:
           X.append(X_train[i])
           y.append(y_train[i])
        #print(y)

        return X, y

    if types=="ind":

        idx=random.choices(ind,k=int(round((len(y_train)*0.5),1)))
        for i in idx:
            X.append(X_data[i])
            y.append(y_data[i])

        return X,y,idx

def generate_csv(dic):
    #grava um arquivo csv com o nome dos atributos na primeira linha, e das instancias, arquivo para script do R
    with open("/media/marcos/Data/Tese/teste.csv",'w') as f:
        w = csv.writer(f)
        w.writerow(dic['class'])
        w.writerows(dic['data'])

def complexity_data():

    proc = subprocess.Popen(['Rscript /home/marcos/Documentos/new_8.r'],
                            stdout=subprocess.PIPE, shell=True)
    (cont_arq, err) = proc.communicate()
    cont_arq = (cont_arq.decode("utf-8"))
   # print(cont_arq)
    cont_arq = cont_arq.split()
    complex = []
   # x=[]
    for i in cont_arq:

        if (i[:3] == "ove" or i[:3] == "nei" or i[:3] == "dim" or i[:3] == "lin" or i[:3] == "bal" or i[:3] == "net"):
            #x.append(i)
            continue
        else:
            if i == "Inf":
                #print(type(i))
                i=0.0
                print("ERROR")
            complex.append(float(i))
   # print(x)
    #exit(0)
    return complex

def complexity_data2(X_data,y_data):
    #comp = []
    dfx = pd.DataFrame(X_data, copy=False)
    dfy = robjects.IntVector(y_data)
    complex = ecol.complexity(dfx, dfy, type="class")

    #print(complex)
    #exit(0)
    complex = np.asarray(complex)
    #print(complex)
    #complex=complex.tolist()
    #exit(0)
    return complex

def complexity_data3(X_data,y_data,grupo,tipo):
    #complex=[]
    dfx = pd.DataFrame(X_data, copy=False)
    dfy = robjects.IntVector(y_data)
    #print(X_data)
    over=nei=line=dim=bal=net=None
    complex = np.array([])
    if grupo[0]=='overlapping':
       # print('entrei')
        over=ecol.overlapping(dfx,dfy,measures=tipo[0])
        over = np.asarray(over)
       # print(over)

        complex = np.append(complex, over[0])
        #print(complex)
        #exit(0)
    if grupo[1]=="neighborhood":
        nei=ecol.neighborhood(dfx,dfy,measures=tipo[1])
        nei = np.asarray(nei)
       # print('entre')
        complex = np.append(complex, nei[0])
    if grupo[2]=="linearity":
        line=ecol.linearity(dfx,dfy,measures=tipo[2])
        line = np.asarray(line)
        complex = np.append(complex, line[0])
    if grupo[3] == "dimensionality":
        #print('entrei')
        dim = ecol.dimensionality(dfx, dfy, measures=tipo[3])
        dim = np.asarray(dim)
        complex = np.append(complex, dim[0])
    if grupo[4] == "balance":
        bal = ecol.balance(dfx, dfy, measures=tipo[4])
        bal = np.asarray(bal)
        complex = np.append(complex, bal[0])
    if grupo[5] == "network":
        net = ecol.network(dfx, dfy, measures=tipo[5])
        net = np.asarray(net)
        complex = np.append(complex, net[0])

    # if over !=None:
    #      over = np.asarray(over)
    #     # print(over)
    #      complex = np.append(complex, over[0])
    # if nei !=None:
    #      nei=np.asarray(nei)
    #      complex = np.append(complex, nei[0])
    # if line !=None:
    #      line=np.asarray(line)
    #      complex = np.append(complex, line[0])
    # #
    # if dim!=None:
    #     dim=np.asarray(dim)
    #     complex = np.append(complex, dim[0])
    # #
    # if bal!=None:
    #      bal=np.asarray(bal)
    #      complex = np.append(complex, bal[0])
    # #
    # if net !=None:
    #      net = np.asarray(net)
    #      complex = np.append(complex, net[0])
    complex=complex.tolist()

    #print (complex)
    #exit(0)
    return complex

def paralell_process(process):
   y=[]
   x=os.popen('Rscript {}'.format(process)).read()
   x=x.split()
  # print(x)
   y.append(x)

   return y

def biuld_dic(X,y, dic):
    #constroi o dicionario para o construir o csv do R (instancias e classes)
    d = dict()
    g = []
    g.append(y)
    g = np.array(g)
    g.astype(int)
    d['class'] = dic['class']
    d['data'] = np.concatenate((X, g.T), axis=1)
    d['data'] = d['data'].tolist()

    return d

def biuld_classifier(X_train, y_train, X_val, y_val):
    #constroi os classificadores, e retorna classificador, score e predict
    perc = perceptron.Perceptron(n_jobs=7, max_iter=100, tol=10.0)
    perc.fit(X_train, y_train)
    score=perc.score(X_val,y_val)
    predict=perc.predict(X_val)

    return perc, score, predict

def dispersion(complexity):

    result=[]

    #y = np.array(complexity)

    dist = pairwise_distances(complexity, n_jobs=6)
    #x=np.indices(dist.diagonal())

    #exit(0)
    dist = dist.tolist()
    for i in dist:
        result.append(np.mean(i))
    return result

def dispersion2(complexity):
   # print(complexity)
    #retorna a dipersao de 1 valor (a-b) entrada ex: ([valor],[valor]....)
    result=[]
    s=[]
    complexity = list(complexity)

    for i in range(len(complexity)):

        t = []
        for j in range(len(complexity)):
            if i==j:
                continue
            else:
                t.append(abs(complexity[i][0] - complexity[j][0]))
        s.append(t)
    for i in s:
        result.append(np.mean(i))

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
               print(q)
        q_test.append(np.mean(q))
        double_faults.append(np.mean(db))

    return q_test, double_faults
def diversity2(y_test,predicts,function):
    div=diversity.compute_pairwise_diversity(y_test,predicts,function)
    return div
def biuld_csv_result(complexity_result,score,Q_test, Df,disp):
    global base_name
    header=['overlapping.F1', 'overlapping.F1v', 'overlapping.F2', 'overlapping.F3', 'overlapping.F4', 'neighborhood.N1', 'neighborhood.N2', 'neighborhood.N3', 'neighborhood.N4', 'neighborhood.T1', 'neighborhood.LSCAvg', 'linearity.L1', 'linearity.L2', 'linearity.L3', '000000.T2', 'dimensionality.T3', 'dimensionality.T4', 'balance.C1', 'balance.C2', 'network.Density', 'network.ClsCoef', 'network.Hubs','Score', 'Q_test','DoubleFault','Disper']
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
        #print('entreivali')
        if (os.path.exists(local+"/Validacao/"+str(iteration)) == False):
            os.system("mkdir -p " + local+"/"+str(iteration))
        with open(local+"/" +str(iteration)+"/"+ base_name + ".csv", 'w') as f:
           # print('entreivali')
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

def oracle(poll,X,y,X_test,y_test):
    orc = Oracle(poll)
    orc.fit(X,y)
    #orc.predict(X_test,y_test)
    return orc.score(X_test,y_test)

def routine_save_bags(local_dataset, local ,base_name, iteration ):

    #rotina para criar treino teste e validaçao alem dos 100 bags, local e onde esta o dataset orig
    X_data, y_data, dataset, dic = open_data(base_name, local_dataset)
    X_train, y_train, X_test, y_test, X_vali, y_vali, id_train, id_test, id_vali = split_data(X_data, y_data)
    #print('mudar as saidas')
    save_bag(id_train, 'train', local+"/Treino/",base_name,(iteration))
    save_bag(id_vali, 'validation', local+"/Validacao/",base_name,str(iteration))
    save_bag(id_test, 'test', local+"/Teste/",base_name,str(iteration))
    for i in range(0, 100):
        #       X_bag, y_bag, id = biuld_bags_stratify(y_train,X_train=X_train, X_data=X_data, y_data=y_data, ind=id_train, types="ind")
        X_bag, y_bag, id = biuld_bags(y_train, X_data=X_data, y_data=y_data, ind=id_train, types="ind")
        #print(len(id))
        id.insert(0,i)
        save_bag(id, 'bags', local+"/Bags/", base_name, str(iteration))
    return  X_train, y_train, X_test, y_test, X_vali, y_vali, dic

def open_bag(local_bag, base_name):
    bags = dict()
    bags['nome'] = list()
    bags['inst'] = list()
    with open(local_bag+base_name+'.csv', 'r') as f:
        reader = csv.reader(f)
        indx = list(reader)
       # print(indx)
    for i in indx:
        bags['nome'].append(i[0])
        bags['inst'].append(i[1:])

    return bags

def biuld_x_y(indx_bag,X,y):

    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
    #global nome_base, classes, caminho_base
    X_data = []
    y_data = []
   # print(indx_bag)
    for i in indx_bag:
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    return X_data, y_data

def open_test_vali(local,base_name,iteration):
   # print("open bag/teste_vali mudar as saidas")
    with open(local + "Teste/"+str(iteration)+"/"+base_name+'.csv','r') as f:
        reader = csv.reader(f)
        teste = list(reader)
    with open(local + "Validacao/"+str(iteration)+"/"+base_name+'.csv','r') as f:
        reader = csv.reader(f)
        vali = list(reader)
    return teste[0], vali[0]

def open_training(local,base_name,iteration):
    with open(local + "Treino/"+str(iteration)+"/"+base_name+'.csv','r') as f:
        reader = csv.reader(f)
        treino=list(reader)
    return treino[0]


def main():
    #import time
    base_name='Wine'
    repeticao="4/"
    local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
    local = "/media/marcos/Data/Tese/Bases3/"
    caminho_base = "/media/marcos/Data/Tese/Bases2/"
    cpx_caminho = "/media/marcos/Data/Tese/Bases3/Bags/"

    X,y,_,_=open_data(base_name,local_dataset)
    test,vali=open_test_vali(local,base_name,repeticao)
    X_test,y_test=biuld_x_y(test,X,y)
    X_vali,y_vali=biuld_x_y(vali,X,y)
    df=[]
    qt=[]
    pred=[]
    bags= open_bag(cpx_caminho+repeticao,base_name)
    for i in bags['inst']:

        X_bag,y_bag=biuld_x_y(i,X,y)
        _,_, pre= biuld_classifier(X_bag,y_bag,X_test,y_test)
        pre=pre.tolist()
        pred.append(pre)


        #pre=pre.split(' ')
        #y_test=np.array(y_test)
        #print(pre)
        #print(y_test)
    pred = np.array(pred)
   # print(len(pre))
    #q,d=diversitys(y_test,pred)
    #df.append(d)
        #qt.append(q)
    print(pred)
    #pred=np.array(pred)
    pred=(pred.T)
    #print(pred)
    a=diversity2(y_test,pred,diversity.Q_statistic)
    print(a)
    #print(df)


    #for i in range(1,21):
   #     X_train, y_train, X_test, y_test, X_vali, y_vali, dic = routine_save_bags(local_dataset, local, "Wine",
    #                                                                              i)

   # x=[]
   # y=[1,2,3,4]
   # x.append(y)
   # x.append(y)
   # k=dispersion2(x)
   # print(k)
    #inicio = time.time()
    #X_data, y_data, dataset, dic = open_data(base_name, local_data)
    #X_train, y_train, X_test, y_test, X_vali, y_vali, id_train, id_test, id_vali = split_data(X_data, y_data)
    #complexity_result=[]
    score=[]
    predict=[]
   # exit(0)

   # print(X_test)

   # for i in range(1,10):
    #print(i)
    #X_bag,y_bag=biuld_bags(y_train=y_train,X_train=X_train,types="sample")
        #print((X_bag))
        #exit(0)
    #newdic=biuld_dic(X_bag,y_bag,dic)
    #generate_csv(newdic)
        #save_bag(id_train,'train',"/media/marcos/Data/Tese/Bases3/Treino",base_name)
    #complexity_result.append(complexity_data())
       # _,sc,pre=biuld_classifier(X_bag,y_bag,X_vali,y_vali)
       # score.append(sc)
       ##oracle(pre,y_vali)

    #q,df=diversitys(y_test,predict)
    #disp = dispersion(complexity_result)
    #biuld_csv_result(complexity_result,score,q,df,disp)
    complexity_result = []
    score = []
    predict = []
    #fim = time.time()
    #print((fim - inicio)/60)
    #print(complexity_result)
    #print((disp))
    #print(len(q))
    #inicio = time.time()
    #process=("/home/marcos/Documentos/new_1.r","/home/marcos/Documentos/new_2.r","/home/marcos/Documentos/new_3.r","/home/marcos/Documentos/new_4.r","/home/marcos/Documentos/new_5.r","/home/marcos/Documentos/new_6.r")
    #pool = Pool(processes=8)
    #y=pool.map(paralell_process,process)
    #print(y)
    #fim = time.time()
    #print((fim - inicio) / 60)
    #inicio = time.time()
    #complexity_data()
    #complexity_data2()
    #fim = time.time()
    #print((fim - inicio) / 60)



if __name__ == "__main__":
    main()



