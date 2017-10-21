import Marff
import os, subprocess
import sys
nome_base = sys.argv[1]

#nome_base = 'Wine'
pasta='Distancias'
#num_class=2
repeticoes=21
temp=""
numero_individuos=101


"""Script responsavel em calcular a complexidade dos arffs"""
"""Cria pastas de acordo com o numero de repeticoes, calcula a complexidade e salva na pasta criada"""

dcol = "/media/marcos/Data/Tese/dcol/DCoL-v1.1/Source/dcol"



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
    elif pasta== 'Distancias':
        dataset = "/media/marcos/Data/Tese/" + pasta +'/'+str(j)+'/' + nome_base + "/Vizinhos"+nome_base+str(j)+'.arff'
        enderecoin = " -i  /media/marcos/Data/Tese/" + pasta +'/'+str(j)+'/' + nome_base #+ "/Vizinhos"+nome_base+str(j)+'.arff'
        proc = subprocess.Popen(
            ["ls /media/marcos/Data/Tese/" + pasta + '/' + str(j) + '/' + nome_base + "/*.arff | wc -l"],
            stdout=subprocess.PIPE, shell=True)
        (cont_arq, err) = proc.communicate()
        cont_arq = int(cont_arq)
        print(cont_arq)
        numero_individuos=cont_arq

    enderecoout = " -o /media/marcos/Data/Tese/Complexidade" + pasta + "/" + nome_base + "/" + nome_base + str(j) + "/complexidade" + nome_base

    if (nome_base != temp):
            temp = nome_base
            dataset= Marff.abre_arff(dataset)
            num_class, classes = Marff.retorna_classes_existentes(dataset)
            #print("numero de classes: ")
            #print (classes)
    for i in range(1,numero_individuos):
        k=i-1
        if(pasta=='Distancias'):
            distancias="/Vizinhos"+nome_base+str(i)
        else:
            distancias='/Individuo'+nome_base+str(i)
        if num_class==2:
            os.system(dcol+enderecoin+distancias+".arff"+enderecoout+str(k)+" -F 1 -N 2")
        else:
            os.system(dcol + enderecoin + distancias + ".arff" + enderecoout + str(k) + " -d -F 1 -N 2")
