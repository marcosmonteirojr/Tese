import  Cpx, Marff, os, sys, csv, numpy as np
from sklearn.externals.joblib import Parallel, delayed
local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
#min_score=0
nome_base='Banana'
#x=sys.argv[2]

#local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
#local = "/home/projeto/Marcos/Bases3"
#caminho_base = "/home/projeto/Marcos/Bases2/"
#cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"
#########################################################################


arq_dataset = local + "Dataset/" + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
X, y, _ = Marff.retorna_instacias(arq_arff)

header=['overlapping.F1', 'overlapping.F1v', 'overlapping.F2', 'overlapping.F3', 'overlapping.F4', 'neighborhood.N1', 'neighborhood.N2', 'neighborhood.N3', 'neighborhood.N4', 'neighborhood.T1', 'neighborhood.LSCAvg', 'linearity.L1', 'linearity.L2', 'linearity.L3', '000000.T2', 'dimensionality.T3', 'dimensionality.T4', 'balance.C1', 'balance.C2', 'network.Density', 'network.ClsCoef', 'network.Hubs']
##exit(0)


def complexidades(y_train,X_train,grupos):
    print("sss")
    X_bag, y_bag = Cpx.biuld_bags(y_train, X_train, types="sample")
    cpx=Cpx.complexity_data4(X_bag,y_bag,grupos)
    del X_bag,y_bag

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

def vote_complexity(X_data,y_data,grupos):
    voto = [0] * 11
    for i in range(1,11):
        print(i)
        if os.path.isfile(local + "/Bags/" + str(i) + "/" + nome_base + ".csv") == False:
            print("Não exite Bags/treino/validation/teste")
            return 0
        stad = []
        comp=[]
        ind=Cpx.open_training(local,nome_base,i)
        X_train, y_train =Cpx.biuld_x_y(ind,X_data,y_data)
        cp=Parallel(n_jobs=4)(delayed(complexidades)(y_train,X_train,grupos) for j in range(100))
        del X_train , y_train
        comp.append(cp)
        cpx=np.squeeze(comp)
        cpx=cpx.T
        np.set_printoptions(threshold=np.nan)
        print(cpx)

       # exit(0)
        for k in cpx:
            norm=Cpx.min_max_norm(k)
            #print(norm)
            stad.append(np.std(norm))
        max=np.argsort(stad)
        #print(stad)
        stad=np.array(stad)
       # exit(0)
        max=max[::-1]
        # with open("/media/marcos/Data/Tese/Resultados_Complexit" + nome_base + ".csv", 'a') as f:
        #
        #     w = csv.writer(f)
        #    # w.writerows(["x"])
        #     w.writerows(cpx)
        #   #  w.writerows(["x"])
        #     #w.writerows(k)
        #    # w.writerows(stad)

        del cpx
        overlapping=stad[0:5]
        #print(len(stad))
        neighborhood=stad[5:11]
        # linearity=stad[11:14]
        # dimensionality=stad[14:17]
        # balance=stad[17:19]
        # network=stad[19:22]
        o=np.argmax(overlapping)
        nei=np.argmax(neighborhood)
        # lin=np.argmax(linearity)
        # dim=np.argmax(dimensionality)
        # bal=np.argmax(balance)
        # net=np.argmax(network)
        voto[o]=voto[o]+1
        voto[nei+5]=voto[nei+5]+1

        #exit(0)
        # voto[lin+11]=voto[lin+11]+1
        # voto[dim+14]=voto[dim+14]+1
        # voto[bal+17]=voto[bal+17]+1
        # voto[net+19]=voto[net+19]+1
        text = ''
        for carro, cor in zip(header, voto):
            text += '{} {}, '.format(carro, cor)
        #print(text)
        print("\n",voto)
        #exit(0)
    return voto, text, max, stad






grupos=["overlapping", 'neighborhood', '', '', '', '']
voto,text, max, std=vote_complexity(X,y,grupos)


arq = open('Voto.txt', 'a')
arq2 = open('Std.txt', 'a')

arq.write(nome_base+" ")
for i in voto:
#
     arq.write(str(i)+" ")
#
arq.write("\n")
arq2.write("std ")
for i in std:
     arq2.write(str(i) + " ")
arq2.write("\n")
arq.close()
arq2.close()
# print(voto)
# # print(comp[0][0])a
# # print((comp[0][1]))
# s=np.squeeze(comp)
# .s=s.tolist()
# print(s[0])
# print(s[1])
# print(s)pr
#print(np.reshape(comp,(200,-1)))
