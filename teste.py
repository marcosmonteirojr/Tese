import numpy as np
import Cpx
file = open('DistdiverP2', "r")

linhas=file.readlines()

matrix2=[]

for i in linhas:
    print(i)
    x=i.split("\t")
    matrix = []
    for j in x:
        matrix.append(float(j))
    matrix2.append(matrix)

cont =1
inicio=0
fim=100
dist=[]
while cont != 21:
    print((cont))
    dist.append(Cpx.dispersion(matrix2[inicio:fim]))
    cont=cont+1
    inicio=fim
    fim=fim+100
resultado=[]
for i in dist:
    resultado.append(np.mean(i))
print(resultado)
print(len(resultado))
exit(0)