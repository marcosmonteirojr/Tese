import numpy as np
import csv
bag_acc=open("/media/marcos/Data/Tese/Resultados/Resultados_novembro2019/resultado final 24-06-2020/teste tabelas/Resultado_acc_bag.csv","r")
ada_acc=open("/media/marcos/Data/Tese/Resultados/Resultados_novembro2019/resultado final 24-06-2020/teste tabelas/Resultado_acc_ada.csv","r")
rf_acc=open("/media/marcos/Data/Tese/Resultados/Resultados_novembro2019/resultado final 24-06-2020/teste tabelas/Resultado_acc_rf.csv","r")
bag_dist=open("/media/marcos/Data/Tese/Resultados/Resultados_novembro2019/resultado final 24-06-2020/teste tabelas/Resultado_dist_bag.csv","r")
ada_dist=open("/media/marcos/Data/Tese/Resultados/Resultados_novembro2019/resultado final 24-06-2020/teste tabelas/Resultado_dist_ada.csv","r")
rf_dist=open("/media/marcos/Data/Tese/Resultados/Resultados_novembro2019/resultado final 24-06-2020/teste tabelas/Resultado_dist_rf.csv","r")

bag_acc1=bag_acc.readlines()
ada_acc1=ada_acc.readlines()
rf_acc1=rf_acc.readlines()
bag_dist1=bag_dist.readlines()
ada_dist1=ada_dist.readlines()
rf_dist1=rf_dist.readlines()
bacc=[]
acc=[]
racc=[]
bdist=[]
adist=[]
rdist=[]
for i in bag_acc1:
    bacc.append(i[:-1].split(";"))
for i in ada_acc1:
    acc.append(i[:-1].split(";"))
for i in rf_acc1:
    racc.append(i[:-1].split(";"))
for i in bag_dist1:
    bdist.append(i[:-1].split(";"))
for i in ada_dist1:
    adist.append(i[:-1].split(";"))
for i in rf_dist1:
    rdist.append(i[:-1].split(";"))
bacc=np.array(bacc[2:])
acc=np.array(acc[2:])
racc=np.array(racc[2:])
bdist=np.array(bdist[2:])
adist=np.array(adist[2:])
rdist=np.array(rdist[2:])

bacc=bacc.T
acc=acc.T
racc=racc.T
bdist=bdist.T
adist=adist.T
rdist=rdist.T


f=[]
f2=[]
f3=[]
for i in range(2,15,2):
    f.append(bdist[i])
    f.append(bacc[i])
    f.append(bacc[i-1])
    f.append(acc[i-1])
    f.append(racc[i-1])



for i in f:
    f2=[]
    for j in i:
        f2.append(j[:-6])
    print(f2)
    f3.append(f2)
f=np.array(f)
f3=np.array(f3)
f=f.T
f3=f3.T


with open('/media/marcos/Data/Tese/Resultados/Resultados_novembro2019/resultado final 24-06-2020/teste tabelas/eggs.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';')
    spamwriter.writerows(f)

with open('/media/marcos/Data/Tese/Resultados/Resultados_novembro2019/resultado final 24-06-2020/teste tabelas/eggs2.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';')
    spamwriter.writerows(f3)
