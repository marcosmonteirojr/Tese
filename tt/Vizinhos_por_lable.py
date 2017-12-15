import arff, os, sys, numpy as np
from scipy.spatial import distance
from datetime import datetime
import Marff as marff
#
'''abre arquivos valida e teste, calcula as distancias, cria bags com a distancia e calcula a complexidade'''



#caminho_vi="/media/marcos/Data/Tese/ComplexidadeDist/" #caminho geral
caminho_te="/media/marcos/Data/Tese/Bases/Teste/"#caminho teste
caminho_va='/media/marcos/Data/Tese/Bases/Validacao/'#caminho calidacao
#caminho_in_dcol=" -i /media/marcos/Data/Tese/Distancias/ResultadosDistanciasValidaTeste/"#caminho entrada dcol
caminho_resultados= "/media/marcos/Data/Tese/Distancias/"#caminho para saida resultados

dcol = "/home/marcos/Documents/Tese/dcol/DCoL-v1.1/Source/dcol"

#nome_b = sys.argv[1]#nome da base
nome_b= 'Glass'
arquivo_t = "Teste"+nome_b #nome dos arquivos
arquivo_v = 'Valida'+nome_b #nome dos arquivos valida

# nome_b = 'Wine'#nome da base
# arquivo_t = "TesteWine" #nome dos arquivos
# arquivo_v = 'ValidaWine' #nome dos arquivos valida
num_vizinhos=30







def cria_arff(info, data, nome):
    """
    cria um arquivo arff no E:
    @:param info: descricao da base/ relacao
    @:param data: dados da base
    @:param nome: do arquivo a ser gerado
    :return:
    """
    obj = {
        'description': info['description'],
        'relation': info['relation'],
        'attributes': info['attributes'],
        'data': data['data'],

    }
    arq1=arff.dumps(obj)
    arq=open('/media/marcos/Data/Tese/Distancias/'+str(i)+'/'+nome_b+'/'+nome+'.arff','w')
    arq.write(arq1)
    arq.close()

def distancia_maxima(data, data2, pos, dist):
    """compara data[pos] com todos data2 1 a um retorna a distancia escolhida
        @:param data: base extraida do arrf
        @:param data2: base a ser medida (arff)
        @:param pos: posicao a ser comparada
        @:param dist: posicao do vetor que fica a distancia

    """
    distancias = []#retornoda distancia
    vetor = (data['data'][pos])
    vetor = (vetor[:-1])
    for j in data2['data']:  # percorre a base valida
        vetor2 = (j[:-1])  # elimina a ultima coluna
        c = distance.euclidean(vetor, vetor2)  # calcula as distancias
        distancias.append(c)  # salva em um array as distancias
    distancias.sort() #ordena
    return distancias, distancias[dist] #retorna o valor da distancia na posisao desejada

def cria_pasta(i):

    global nome_base
    nome_base = nome_b + str(i)
    dataset=(caminho_te + str(i) + '/' + arquivo_t + str(i) + '.arff')
    dataset2 = (caminho_va + str(i) + '/' + arquivo_v + str(i) + '.arff')

    # if (os.path.exists(caminho_vi + nome_b) == False):
    #     os.system("mkdir "+caminho_vi+ nome_b)
    # if (os.path.exists(caminho_vi + nome_b+"/"+nome_base) == False):
    #     os.system("mkdir "+caminho_vi + nome_b + "/" + nome_base)
    if(os.path.exists(caminho_resultados+str(i))==False):
        os.system("mkdir " +caminho_resultados + str(i))
    if (os.path.exists(caminho_resultados + str(i)+'/'+nome_b) == False):
        os.system("mkdir " + caminho_resultados + str(i)+'/'+nome_b)

    return  dataset, dataset2

def abre_arff(dataset, dataset2):

    teste = marff.abre_arff(dataset)
    X_teste, y_teste = marff.retorna_instacias(teste)
    validacao = marff.abre_arff(dataset2)
    X_valida, y_valida = marff.retorna_instacias(validacao)
    return X_teste, y_teste, X_valida, y_valida, teste, validacao

def lista_vizinhos(num_viz, num_c, elementos_p_classes=None, total_elementos=None):
    '''

    :param num_viz: numero de vizinhos
    :param num_c: numero de classes
    :return: lista com o numero de elementos dividido
    '''


    cont=0
    vet=[0]*num_c
    while cont != num_viz:
        for i in range(0, num_c):
            if cont == num_viz:
                break
            else:
                vet[i] += 1
                cont += 1
    if elementos_p_classes != None and total_elementos!=None:

        for j in range(len(vet)):
            x=elementos_p_classes[j]/total_elementos
            vet[j]=(x*num_viz)
            if(vet[j]<1):
                vet[j]=1
            vet[j]=int(round(vet[j],0))
    return vet


#(nome_b)
dataset, dataset2 = cria_pasta(1)
X_teste, y_teste, X_valida, y_valida, teste, validacao = abre_arff(dataset, dataset2)
num_classes, todas_as_classes, elementos_p_classes, total_elementos = marff.retorna_classes_existentes(teste)
#get_vizinhos = num_vizinhos / num_classes
print(todas_as_classes, elementos_p_classes,total_elementos)

lista_n_vizinhos = lista_vizinhos(num_vizinhos,num_classes,elementos_p_classes,total_elementos)
print(lista_n_vizinhos)
#todas_as_classes.sort()
'''colocar as classes de 0 a N'''
x=[str(i) for i in todas_as_classes]
teste['attributes'][-1]=('Class',x)

#print(nome_b)

#csv=open("Vizinhacas2.csv",'a')
now=datetime.now()
#csv.write('Base;Data e hora;Numero de vizinhos; Numeros de bootstraps; Termino\n')
#csv.write(str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'-'+str(now.hour)+':'+str(now.minute)+';'+nome_b+';'+str(num_vizinhos)+';'+str(len(X_teste))+'\n')

# for xx in range(num_classes):
#     lista_n_vizinhos.append(get_vizinhos)
lista_teste=[]
lista_teste2=[]
for p in range(len(X_teste)):
    lista_teste2.append(p)
for i in range(1, 21):

    dataset, dataset2 = cria_pasta(i)
    X_teste, y_teste, X_valida, y_valida, teste, validacao = abre_arff(dataset, dataset2)
    #todas_as_classes.sort()
    x = [str(i) for i in todas_as_classes]
    teste['attributes'][-1] = ('Class', x)
    lista_teste = []
    for l in range(len(X_teste)):
        dados = dict()
        dados['data'] = list()
        lista_distancias=[]
        lista_c=[]
        lista_n_vizinhos = lista_vizinhos(num_vizinhos, num_classes,elementos_p_classes,total_elementos)

        for j in X_valida:
            c = distance.euclidean(X_teste[l], j)
            lista_distancias.append(c)
        indices_ordenados = np.argsort(lista_distancias)
        cont=0
        for yy in X_valida:
            lista_c.append(yy[:])

        for k in indices_ordenados:
            x=y_valida[k]
            if(lista_n_vizinhos[x]!=0):
                lista_n_vizinhos[x]+=-1
                lista_c[k].append(y_valida[k])
                dados['data'].append(lista_c[k])
                cont+=1
                cont1 = lista_n_vizinhos.count(0)
                #print(len(indices_ordenados))
            if (cont1 == num_classes):
                lista_teste.append(l)
                if(lista_teste==lista_teste2):
                    print(nome_b, lista_n_vizinhos, l, i)

            if cont==num_vizinhos:

                #print(lista_n_vizinhos)
                break

        vizinhos = 'Vizinhos' + nome_b + str(l)
        cria_arff(teste, dados, vizinhos)
now=datetime.now()
#csv.write(str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'-'+str(now.hour)+':'+str(now.minute)+'\n\n')
#csv.closed



        #for zz in range(len(X_valida)):
        #print(validacao['data'][zz])
        #csv.write(';;;{d[0]};{d[1]};{d[2]};{c};;{g}\n'.format(d=validacao['data'][zz],c=lista_distancias[zz],g=indices_ordenados[zz]))
        #print(*X_valida[0],sep=';')
        #print('{d[0]}{d[1]};{};{};{};\n'.format(d=X_teste[l], y_teste[l], X_valida, y_valida))
        #exit(0)


