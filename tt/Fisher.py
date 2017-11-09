import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import Marff
from scipy.spatial import distance
base=Marff.abre_arff('/media/marcos/Data/Tese/Ag/1/IndividuoAdult1.arff')
X,y=Marff.retorna_instacias(base)
#print(X[0])

i1 = []
i2 = []
fisher=[]
g=[]
gg=[]
n2=[]
n22=[]
inter=0
intra=0
# for i in range(len(y)):
#     if y[i]==y[0]:
#         i1.append(X[i])
#     else:
#         i2.append((X[i]))

# for j in range(len(i1)):
#     for k in range(len(i2)):
#         v=(abs((np.mean(i1[j]) - np.mean(i2[k]))**2)/((np.var(i1[j]) + np.var(i2[k]))))
#         #print(v)
#         fisher.append(v)
#print(max(fisher))

dist=0

for j in range(len(X)):
    min_intra = 999999999
    min_inter = 999999999
    for k in range(len(X)):
        if(X[j]==X[k]):
            continue
        else:

           dist=distance.euclidean(X[j],X[k],)
        if y[j]!=y[k] and dist<min_inter:
            min_inter=dist
        if y[j]==y[k] and dist<min_intra:
            min_intra=dist
    inter+=min_inter
    intra+=min_intra
    print(inter, intra)
print(intra/inter)

# for j in range(len(X)):
#     min_intra = []
#     min_inter = []
#     for k in range(len(X)):
#         if(X[j]==X[k]):
#             continue
#         else:
#            dist=distance.euclidean(X[j],X[k])
#         if y[j]!=y[k]:
#             min_inter.append(dist)
#         if y[j]==y[k]:
#             min_intra.append(dist)
#     a=min(min_inter)
#     b=min(min_intra)
#     c=max(min_inter)
#     d=max(min_intra)
#
#     a=a-c
#     b=b-d
#
#     inter+=a
#     intra += b
#     print(inter, intra)
# print(intra/inter)
#print(sum(n2))

