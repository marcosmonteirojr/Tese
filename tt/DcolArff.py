import Marff
import os, subprocess
from datetime import datetime
import sys
#nome_base = sys.argv[1]

nome_base = 'Adult'
pasta='Ag'
repeticoes=21
temp=""
numero_individuos=101
caminho='/media/marcos/Data/Tese/Complexidade'
caminho_in='/media/marcos/Data/Tese/'

"""Script responsavel em calcular a complexidade dos arffs"""
"""Cria pastas de acordo com o numero de repeticoes, calcula a complexidade e salva na pasta criada"""

dcol = "/media/marcos/Data/Tese/dcol/DCoL-v1.1/Source/dcol"
csv=open("Dcol.csv",'a')
now=datetime.now()
csv.write('Data e hora;Base;Tipo;Termino\n')
csv.write(str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'-'+str(now.hour)+':'+str(now.minute)+';{};{};\n'.format(nome_base,pasta))



for j in range(1, repeticoes):
    if (os.path.exists(caminho + pasta) == False):
        os.system("mkdir "+caminho + pasta)
    if (os.path.exists(caminho + pasta+ "/" + nome_base) == False):
        os.system("mkdir "+caminho + pasta + "/" + nome_base)
    if (os.path.exists(caminho + pasta + "/" + nome_base + "/" + nome_base + str(j)) == False):
        os.system("mkdir "+caminho + pasta + "/" + nome_base + "/" + nome_base + str(j))

    if pasta == 'Bags':
        dataset = caminho_in+pasta+'/'+str(j)+'/Individuo'+nome_base+str(1)+'.arff'
        enderecoin = " -i "+caminho_in+pasta+'/' + str(j)
    elif pasta == 'Ag':
        dataset = caminho_in+pasta+'/' +str(j)+'/Individuo'+nome_base+str(1)+'.arff'
        enderecoin = " -i "+caminho_in+pasta+'/' + str(j)
    elif pasta== 'Distancias':
        dataset = caminho_in + pasta +'/'+str(j)+'/' + nome_base + "/Vizinhos"+nome_base+str(j)+'.arff'
        enderecoin = " -i  "+caminho_in + pasta +'/'+str(j)+'/' + nome_base #+ "/Vizinhos"+nome_base+str(j)+'.arff'
        proc = subprocess.Popen(
            ["ls /media/marcos/Data/Tese/" + pasta + '/' + str(j) + '/' + nome_base + "/*.arff | wc -l"],
            stdout=subprocess.PIPE, shell=True)
        (cont_arq, err) = proc.communicate()
        cont_arq = int(cont_arq)
        #

        print(cont_arq)

        numero_individuos=cont_arq

    enderecoout = " -o /media/marcos/Data/Tese/Complexidade" + pasta + "/" + nome_base + "/" + nome_base + str(j) + "/complexidade" + nome_base

    if (nome_base != temp):
            temp = nome_base
            dataset= Marff.abre_arff(dataset)
            num_class, classes = Marff.retorna_classes_existentes(dataset)
            #print("numero de classes: ")
            #print (classes)
    if (pasta == 'Distancias'):
        inicio=0
    else:
        inicio=1
    for i in range(inicio,numero_individuos):

        if(pasta=='Distancias'):
            distancias="/Vizinhos"+nome_base+str(i)
            print(distancias)
            k=i

        else:
            distancias='/Individuo'+nome_base+str(i)
            k = i-1
        if num_class==2:
            (os.system(dcol+enderecoin+distancias+".arff"+enderecoout+str(k)+" -F 1 -N 2"))

        else:
            os.system(dcol + enderecoin + distancias + ".arff" + enderecoout + str(k) + " -d -F 1 -N 2")

now=datetime.now()
csv.write(str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'-'+str(now.hour)+':'+str(now.minute)+'\n\n')
csv.closed


