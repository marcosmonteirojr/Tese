import  Cpx, Marff, os, numpy as np
from sklearn.externals.joblib import Parallel, delayed
local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
#min_score=0
nome_base='Wine'
#local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
#local = "/home/projeto/Marcos/Bases3"
#caminho_base = "/home/projeto/Marcos/Bases2/"
#cpx_caminho="home/projeto/Marcos/Bases3/Bags/"
for repeticao in range(1,21):
    if os.path.isfile(local + "/Bags/" + str(repeticao) + "/" + nome_base + ".csv") == False:
        print(repeticao)
        ##################Criar bags############################################
        X_train, y_train, X_test, y_test, X_vali, y_vali, dic = Cpx.routine_save_bags(local_dataset, local, nome_base,
                                                                                      repeticao)
#########################################################################


arq_dataset = caminho_base + "Dataset/" + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
X, y, _ = Marff.retorna_instacias(arq_arff)
header=['overlapping.F1', 'overlapping.F1v', 'overlapping.F2', 'overlapping.F3', 'overlapping.F4', 'neighborhood.N1', 'neighborhood.N2', 'neighborhood.N3', 'neighborhood.N4', 'neighborhood.T1', 'neighborhood.LSCAvg', 'linearity.L1', 'linearity.L2', 'linearity.L3', '000000.T2', 'dimensionality.T3', 'dimensionality.T4', 'balance.C1', 'balance.C2', 'network.Density', 'network.ClsCoef', 'network.Hubs']
##exit(0)


def complexidades(y_train,X_train):
   # print(y_train)
    X_bag, y_bag = Cpx.biuld_bags(y_train, X_train, types="sample")
    cpx=Cpx.complexity_data2(X_bag,y_bag)

    return cpx

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def parallel_distance(i,bags,X,y,X_vali,y_vali):

    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = Cpx.biuld_x_y(indx_bag1,X,y)
    cpx=(Cpx.complexity_data2(X_bag, y_bag))
   # _, score, pred = Cpx.biuld_classifier(X_bag, y_bag, X_vali, y_vali)

    return cpx #score, pred

def vote_complexity(X_data,y_data):
    voto = [0] * 22
    for i in range(1,11):
        print(i)
        if os.path.isfile(local + "/Bags/" + str(i) + "/" + nome_base + ".csv") == False:
            print("Não exite Bags/treino/validation/teste")
            return 0
        stad = []
        comp=[]
        #ind=Cpx.open_training(local,nome_base,i)
        #X_train, y_train =Cpx.biuld_x_y(ind,X_data,y_data)
        # cp=Parallel(n_jobs=6)(delayed(complexidades)(y_train,X_train) for j in range(100))
        _,vali=Cpx.open_test_vali(local,nome_base,i)
        X_vali,y_vali=Cpx.biuld_x_y(vali,X_data,y_data)
        bags=Cpx.open_bag(local + "/Bags/" + str(i) + "/",nome_base)
        cp = Parallel(n_jobs=6, verbose=5)(delayed(parallel_distance)(j, bags,X_data,y_data,X_vali,y_vali) for j in range(len(bags['nome'])))
        #cp, ac, pre = zip(*r)
        #print(len(ac))
        #print(len(pre))
        #print(len(cp))
        #exit(0)
        comp.append(cp)

        #comp=np.append(comp,cp)
        cpx=np.squeeze(comp)
        cpx=cpx.T
        #print(comp)
        for k in cpx:
            #print(len(k))
            stad.append(np.std(k))
        max=np.argsort(stad)
        max=max[::-1]

        overlapping=stad[0:5]
        neighborhood=stad[5:11]
        linearity=stad[11:14]
        dimensionality=stad[14:17]
        balance=stad[17:19]
        network=stad[19:22]
        o=np.argmax(overlapping)
        nei=np.argmax(neighborhood)
        lin=np.argmax(linearity)
        dim=np.argmax(dimensionality)
        bal=np.argmax(balance)
        net=np.argmax(network)
        voto[o]=voto[o]+1
        voto[nei+5]=voto[nei+5]+1
        voto[lin+11]=voto[lin+11]+1
        voto[dim+14]=voto[dim+14]+1
        voto[bal+17]=voto[bal+17]+1
        voto[net+19]=voto[net+19]+1
        text = ''
        for carro, cor in zip(header, voto):
            text += '{} {}, '.format(carro, cor)
        #print(text)
    return voto, text, max, stad

voto,text, max, std=vote_complexity(X,y)
for i in max:
    print(header[i])
print(std)
#print(voto)
# print(comp[0][0])a
# print((comp[0][1]))
# s=np.squeeze(comp)
# .s=s.tolist()
# print(s[0])
# print(s[1])
# print(s)pr
#print(np.reshape(comp,(200,-1)))
