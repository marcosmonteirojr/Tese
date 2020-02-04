import  Cpx, Marff, os, sys, csv, numpy as np
from joblib import  Parallel, delayed
local_dataset = "/media/marcos/Data/Tese/Bases4/Dataset/"
local = "/media/marcos/Data/Tese/Bases4/"
caminho_base = "/media/marcos/Data/Tese/Bases4/"
cpx_caminho="/media/marcos/Data/Tese/Bases4/Bags/"
#min_score=0
##############33jonathan########################
nome_base='ds'
#noe_base_test='features_dsfold1-40-test'
#local="/media/marcos/Data/Tese/Bases4/Dataset/features_tcnn_breakhis"
X,y=Cpx.open_data_jonathan(local_dataset,nome_base)
####################################################
#nome_base=sys.argv[1]

#local_dataset = "/home/marcosmonteiro/Marcos/Bases3/Dataset/"
#local = "/home/marcosmonteiro/Marcos/Bases3/"
#caminho_base = "/home/marcosmonteiro/Marcos/Bases3/"
#cpx_caminho="/home/marcosmonteiro/Marcos/Bases3/Bags/"
#########################################################################


#arq_dataset = local + "Dataset/" + nome_base + ".arff"
#arq_arff = Marff.abre_arff(arq_dataset)
#X, y, _ = Marff.retorna_instacias(arq_arff)
_,val=Cpx.open_test_vali(local,nome_base,1)
X_val, y_val= Cpx.biuld_x_y(val,X,y)

###########################################################################
header=['overlapping.F1', 'overlapping.F1v', 'overlapping.F2', 'overlapping.F3', 'overlapping.F4', 'neighborhood.N1', 'neighborhood.N2', 'neighborhood.N3', 'neighborhood.N4', 'neighborhood.T1', 'neighborhood.LSCAvg', 'linearity.L1', 'linearity.L2', 'linearity.L3', '000000.T2', 'dimensionality.T3', 'dimensionality.T4', 'balance.C1', 'balance.C2', 'network.Density', 'network.ClsCoef', 'network.Hubs']
header1=[['overlapping.F1', 'diver'], ['overlapping.F1v', 'diver'], ['overlapping.F2', 'diver'], ['overlapping.F3', 'diver'], ['overlapping.F4', 'diver'], ['neighborhood.N1', 'diver'], ['neighborhood.N2', 'diver'], ['neighborhood.N3', 'diver'], ['neighborhood.N4', 'diver'], ['neighborhood.T1', 'diver'], ['neighborhood.LSCAvg', 'diver']]

##exit(0)
grupos=["overlapping", 'neighborhood', '', '', '', '']


def complexidades(y_train,X_train,grupos):
    #print("sss")
    X_bag, y_bag = Cpx.biuld_bags(y_train, X_train, types="sample")
    cpx=Cpx.complexity_data4(X_bag,y_bag,grupos)
    #_,acc,pred=Cpx.biuld_classifier(X_bag,y_bag,X_val,y_val)
    #diver = Cpx.diversitys(y_val,pred)
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
       # np.set_printoptions(threshold=np.nan)
        comp.append(cp)
        cpx=np.squeeze(comp)
        cpx=cpx.T

        for k in cpx:
            norm=Cpx.min_max_norm(k)
           # print(k)
            std=np.std(norm)
            #print(std)
            std=std.tolist()
            print(std)
            stad.append(std)
       # exit(0)

        max=np.argsort(stad)
        stad=np.array(stad)
        max=max[::-1]


        del cpx
        overlapping=stad[0:5]
        neighborhood=stad[5:11]

        o=np.argmax(overlapping)
        nei=np.argmax(neighborhood)

        voto[o]=voto[o]+1
        voto[nei+5]=voto[nei+5]+1

        text = ''
        for carro, cor in zip(header, voto):
            text += '{} {}, '.format(carro, cor)
        #print(text)
        print("\n",voto)
        #exit(0)
    return voto, text, max, stad

def salvar_complexidades(X_data,y_data,grupos):
    ind = Cpx.open_training(local, nome_base, 1)
    X_train, y_train = Cpx.biuld_x_y(ind, X_data, y_data)
    r = Parallel(n_jobs=4)(delayed(complexidades)(y_train, X_train, grupos) for j in range(100))
    c, score, pred = zip(*r)
    diver=Cpx.diversitys(y_val, pred)
    print(c)
    print(score)
    print(diver)
    import csv
    with open(nome_base+'correladiver.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile,)
        for i in c:
            spamwriter.writerow(i)
        spamwriter.writerow(score)
        spamwriter.writerow(diver)

    return c, score, diver

def vote_jonathan(local,grupos,kkk=False):
    voto = [0] * 11

    for i in range(1,2):
        print(i)
        if kkk:
            return 0
        stad = []
        comp = []
        nome_base = 'features_dsfold'+str(i)+'-40-test.txt'
        X_train, y_train = Cpx.open_data_jonathan(local, nome_base)
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        cp = Parallel(n_jobs=4)(delayed(complexidades)(y_train, X_train, grupos) for j in range(100))
        del X_train, y_train
        np.set_printoptions(threshold=np.nan)
        comp.append(cp)

        cpx = np.squeeze(comp)
        cpx = cpx.T
        for k in cpx:
            norm = Cpx.min_max_norm(k)
            # print(k)
            std = np.std(norm)
            # print(std)
            std = std.tolist()
            print(std)
            stad.append(std)
            # exit(0)

        max = np.argsort(stad)
        stad = np.array(stad)
        max = max[::-1]
        del cpx
        overlapping = stad[0:5]
        neighborhood = stad[5:11]

        o = np.argmax(overlapping)
        nei = np.argmax(neighborhood)

        voto[o] = voto[o] + 1
        voto[nei + 5] = voto[nei + 5] + 1

        text = ''
        for carro, cor in zip(header, voto):
            text += '{} {}, '.format(carro, cor)
        # print(text)
        print("\n", voto)
    return voto, text, max, stad
####################
grupos=["overlapping", 'neighborhood', '', '', '', '']


import time
ini=time.time()
voto,text, max, std=vote_complexity(X,y,grupos)
#voto,texti, max, std=vote_jonathan(local,grupos)
fim=time.time()
print(fim-ini)
exit(0)
arq = open('Voto.txt', 'a')
arq2 = open('Std.txt', 'a')

arq.write(+" ")
for i in voto:
      arq.write(str(i)+" ")
arq.write("\n")
arq2.write("std ")
for i in std:
      arq2.write(str(i) + " ")
arq2.write("\n")
arq.close()
arq2.close()

###################################3
# print(voto)
# # print(comp[0][0])a
# # print((comp[0][1]))
# s=np.squeeze(comp)
# .s=s.tolist()
# print(s[0])
# print(s[1])
# print(s)pr
#print(np.reshape(comp,(200,-1)))
#import Graficos_ga as g
#c,_,diver=salvar_complexidades(X,y,grupos=grupos)
#c=np.array(c)
#c1=c.T
#for i in range (len(c1)):
#    g.grafico_disper(nome_base, header1[i], c1[i], diver, i ,1, "correlacao", valor3=None,  legend=None)
