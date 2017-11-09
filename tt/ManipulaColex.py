import subprocess, os, sys
import csv, Marff
from datetime import datetime



nome_base = "Wine"


def cria_resumo(pasta,i,out,nome_base):

    """
    junta todos resultados das complexidade e grava em arquivo resumo na pasta destino
    :param pasta: pasta destino
    :param i: caminho entre as pastas ex. Wine+i = Wine1
    :param out: Quantidade de arquivos na pasta
    :param nome_base:
    :return:
    """

    arq2 = open(pasta + nome_base + str(i) + '/' + nome_base + '_resumo.txt', 'w')  # arquivo que salva os resumos

    for j in range(out):
       # if tipo == 0:
        arq1 = open(pasta + nome_base+str(i) + '/complexidade' + nome_base + str(j) + '.txt', 'r')  # abre os aquivos um por um da comlexidade
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        if (j==0):  # filtra o heder e cria um cmo ' F1 e N2"
            for y in range(nclas+1):
                arq2.write(arq1.readline())
        arq1.readline()
        for p in range(nclas):
            arq2.write(arq1.readline())  # salva valores de F1 e N2, cuidar com a qntidade de classes
        #arq2.write(arq1.readline())
        #arq2.write(arq1.readline())
        arq2.write("\n")
        arq1.close()

    arq2.close()


def cria_csv(pasta,i,nome_base):

    """
    trasnforma o resumo em csv
    idem ao cria resumo
    :param pasta:
    :param i:
    :param nome_base:
    :return:
    """

    arq3 = open(pasta + nome_base+str(i) + '/' + nome_base + '_resumo.txt',
            'r')  # abre o resumo que foi gerado acima
    arq4 = open(pasta + nome_base+str(i) + '/' + nome_base + '_resumo2.csv',
            'w')  # salva um novo arquivo formatado em csv


    for i in (arq3):
        i = i.replace('                               ', ',')  # 9
        i = i.replace('          ', ',')  # 10
        i = i.replace('         ', ',')  # 9
        i = i.replace('        ', ',')  # 8
        i = i.replace('       ', ',')  # 7
        i = i.replace('      ', ',')  # 6
        i = i.replace('     ', ',')  # 5
        i = i.replace('   ', ',')  # 3

        # print (i)
        arq4.write(i)
    arq3.close()
    arq4.close()


def calcula_media(pasta, nome_base, i, nclas, d):
    """
    calcula a media de acordo com o numero de classes, arquivo csv
    :param pasta:
    :param nome_base:
    :param i:
    :param nclas: Numero de classes
    :param d: Divisao
    :return:
    """
    rownum =0
    cont=cont2 = 0.0
    media = []  # medias de F1
    median2 = []  # medias de F2
    f1=n2=0
   
    
    arq5 = open(pasta + nome_base+str(i) + '/' + nome_base + '_medias.txt',
                'w')
    # salva um arquivo com as medias, varia de acordo com o numero de classes
    print(pasta + nome_base+str(i) + '/' + nome_base + '_resumo2.csv')
    with open(pasta + nome_base+str(i) + '/' + nome_base + '_resumo2.csv','r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')


        for row in spamreader:

            if rownum == 0:
                None
            else:
                colnum = 0
                # percorre F1 e calcula a media, percorre N2 e calcula a media levando em conta se possui infinito ou nao. lembrar de trocar o cont de acordo com o numero de classes
                for col in row:
                    if (cont <= nclas and colnum == 1):
                        f1 += (float(col))
                        print(f1)
                        #print('\n')
                        cont += 1
                        if(nclas!=2):
                            if cont == nclas:
                                f1 = f1 / d
                                #

                                # print(f1)
                                media.append(f1)
                                cont=0
                                f1 = 0
                        else:
                            media.append(f1)
                            cont = 0
                            f1 = 0
                    if (cont2 <= nclas and colnum == 2):
                        n2 += (float(col))
                        cont2 += 1
                        if(nclas!=2):
                            if cont2 == nclas:
                                n2 = n2 / d
                                median2.append(n2)
                                n2 = 0
                                cont2=0
                        else:
                            median2.append(n2)
                            n2 = 0
                            cont2 = 0
                            #print (median2)
                    colnum += 1
            rownum += 1
        test = map(list, zip(media, median2))  # compacta e transfora em uma matriz o resultado das medias
        arq5.write('F1     N2\n')
        te = [str(i) for i in test]  # converte para string
        for i in te:
            i.split(',', 1)
        for i in te:
            arq5.write(i[1:-1] + "\n")  # salva em um arquivo
    arq5.close()
    csvfile.close()


def diretorios(tipo,nome_base):
    """
    Cria caminho para as pastas
    :param tipo: Caminho 1 pastas das comlp dos bags, 2 das distancias, 3 GA
    :param nome_base:
    :return: pasta, e numeros de arquivos da pasta 1
    """
    if tipo ==1:
        pasta = ('/media/marcos/Data/Tese/ComplexidadeBags/'+nome_base+'/')

    elif tipo==2:
        pasta =('/media/marcos/Data/Tese/ComplexidadeDistancias/'+nome_base+'/')

    elif tipo ==3:
        pasta = ('/media/marcos/Data/Tese/ComplexidadeAg/'+nome_base+'/')

    proc = subprocess.Popen(["ls "+pasta+"/" + nome_base + "1/comple*.txt | wc -l"],
                            stdout=subprocess.PIPE, shell=True)
    (cont_arq, err) = proc.communicate()
    cont_arq = int(cont_arq)  # retorna o numeros de arquivos na pasta
    #print(cont_arq)
    return pasta, cont_arq





#csv=open("Resumos.csv",'a')
#now=datetime.now()
#csv.write('Data e hora;Base;Tipo;Termino\n')
#csv.write(str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'-'+str(now.hour)+':'+str(now.minute)+';{}\n'.format(nome_base))

pasta, cont_arq = diretorios(3,nome_base)
#print(pasta + "/" + nome_base + str(1) + "/Complexidade" + nome_base + str(1) + ".arff")
dataset=Marff.abre_arff('/media/marcos/Data/Tese/Bases/Teste/1/Teste'+nome_base+str(1)+".arff")
num_classes=Marff.retorna_classes_existentes(dataset)
#print(num_classes[0])


if (num_classes[0]>2):
    nclas = num_classes[0]
   # print(nclas)
    #n=nclas-1
   # cont=0
   # for i in range(n, 0, -1):
        #cont=cont+i

    d = nclas
else:
    nclas = num_classes[0]
    d = 1

#print(cont_arq)
for i in range(1,21):
    cria_resumo(pasta=pasta, i=i, out=cont_arq, nome_base=nome_base)
    cria_csv(pasta=pasta,i=i,nome_base=nome_base)
    calcula_media(pasta=pasta, nome_base=nome_base, i=i, nclas=nclas,d=d)
    os.system(
        "rm "+pasta + nome_base+str(i) + "/" + "*.log")
    #print("/media/marcos/Data/Tese/Complexidade" + pasta + nome_base + "/" )








