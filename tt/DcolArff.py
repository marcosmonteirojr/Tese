import Marff
import os
import sys
nome_base = sys.argv[1]

#nome_base = 'Banana'
pasta='Ag'
num_class=2
repeticoes=11
temp=""
numero_individuos=101


"""Script responsavel em calcular a complexidade dos arffs"""
"""Cria pastas de acordo com o numero de repeticoes, calcula a complexidade e salva na pasta criada"""

dcol = "/home/marcos/Documents/Tese/dcol/DCoL-v1.1/Source/dcol"



for j in range(1, repeticoes):
    if (os.path.exists("/media/marcos/Data/Tese/Complexidade" + pasta) == False):
        os.system("mkdir /media/marcos/Data/Tese/Complexidade" + pasta)
    if (os.path.exists("/media/marcos/Data/Tese/Complexidade" + pasta+ "/" + nome_base) == False):
        os.system("mkdir /media/marcos/Data/Tese/Complexidade" + pasta + "/" + nome_base)
    if (os.path.exists("/media/marcos/Data/Tese/Complexidade" + pasta + "/" + nome_base + "/" + nome_base + str(j)) == False):
        os.system("mkdir /media/marcos/Data/Tese/Complexidade" + pasta + "/" + nome_base + "/" + nome_base + str(j))

    if pasta == 'Bags':
        dataset = "/media/marcos/Data/Tese/Bags/"+str(j)+'/Individuo'+nome_base+str(1)+'.arff'
        enderecoin = " -i /media/marcos/Data/Tese/Bags/" + str(j)
    elif pasta == 'Ag':
        dataset = "/media/marcos/Data/Tese/Ag/" +str(j)+'/Individuo'+nome_base+str(1)+'.arff'
        enderecoin = " -i /media/marcos/Data/Tese/Ag/" + str(j)
    else:
        dataset = "/media/marcos/Data/Tese/" + pasta + nome_base + "/" + nome_base + str(j)
        enderecoin = " -i /media/marcos/Data/Tese/"+pasta + nome_base + "/" + nome_base + str(j)
    enderecoout = " -o /media/marcos/Data/Tese/Complexidade" + pasta + "/" + nome_base + "/" + nome_base + str(j) + "/complexidade" + nome_base

    if (nome_base != temp):
        temp = nome_base
        dataset= Marff.abre_arff(dataset)
        num_class, classes = Marff.retorna_classes_existentes(dataset)
        print("numero de classes: ")
        print (classes)
        exit()
    for i in range(1,numero_individuos):
        k=i-1
        distancias='/Individuo'+nome_base+str(i)
        if num_class==2:
            os.system(dcol+enderecoin+distancias+".arff"+enderecoout+str(k)+" -F 1 -N 2")
        else:
            os.system(dcol + enderecoin + distancias + ".arff" + enderecoout + str(k) + " -d -F 1 -N 2")
