import arff
from scipy.spatial import distance
#
'''abre arquivos valida e teste, calcula as distancias, cria bags com a distancia e calcula a complexidade'''



caminho_vi="/media/marcos/Data/Tese/ComplexidadeDist/" #caminho geral
caminho_te="/media/marcos/Data/Tese/Bases/Teste/"#caminho teste
caminho_va='/media/marcos/Data/Tese/Bases/Validacao/'#caminho calidacao
caminho_in_dcol=" -i /media/marcos/Data/Tese/Distancias/ResultadosDistanciasValidaTeste/"#caminho entrada dcol
caminho_resultados= "/media/marcos/Data/Tese/Distancias/ResultadosDistanciasValidaTeste/"#caminho para saida resultados

dcol = "/home/marcos/Documents/Tese/dcol/DCoL-v1.1/Source/dcol"

nome_b = 'Wine'#nome da base
arquivo_t = "TesteWine" #nome dos arquivos
arquivo_v = 'ValidaWine' #nome dos arquivos valida
num_visinhos=30





def euclidean4(vector1, vector2):
    """

    @:param vector1: dados a serem medidos
    @:param vector2: dados a serem medidos
    @:return: distancia
    """

    dist = distance.euclidean(vector1, vector2)
    return dist

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
    arq=open('/media/marcos/Data/Tese/Distancias/ResultadosDistanciasValidaTeste/'+nome_base+'/'+nome+'.arff','w')
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
        c = euclidean4(vetor, vetor2)  # calcula as distancias
        distancias.append(c)  # salva em um array as distancias
    distancias.sort() #ordena
    return distancias, distancias[dist] #retorna o valor da distancia na posisao desejada

def cria_pasta(i):

    global teste,validacao, dcol,enderecoin, enderecoout

   # dataset = arff.load(open(caminho_te + str(i)+'/' + arquivo_t + str(i) + '.arff'))
   # dataset2 = arff.load(open(caminho_va + str(i)+'/' + arquivo_v + str(i) + '.arff'))

    dataset=(caminho_te + str(i) + '/' + arquivo_t + str(i) + '.arff')
    dataset2 = (caminho_va + str(i) + '/' + arquivo_v + str(i) + '.arff')
    teste=Marff.abre_arff(dataset)
    validacao=Marff.abre_arff(dataset2)


    # enderecoin =  caminho_in_dcol+nome_base + "/"
    # enderecoout = " -o " + caminho_vi + nome_b + "/" + nome_base + "/complexidade" + nome_b
    #
    # if (os.path.exists(caminho_vi + nome_b) == False):
    #     os.system("mkdir "+caminho_vi+ nome_b)
    # if (os.path.exists(caminho_vi + nome_b+"/"+nome_base) == False):
    #     os.system("mkdir "+caminho_vi + nome_b + "/" + nome_base)v
    # if(os.path.exists(caminho_resultados+nome_base)==False):
    #     os.system("mkdir " +caminho_resultados + nome_base)

def main():

    for i in range(1,21):
        global nome_base,k,nome_b#nome_base serve para abrir os arquivos, nome_b para dar nome ao novos arquivos
        nome_base = nome_b+str(i)
        cria_pasta(i)
        num_classes, todas_as_classes = Marff.retorna_classes_existentes(teste)
        get_vizinhos = num_visinhos/num_classes
        #print('Numero de classes {}, numero de vizinhos {}, numero vizinhos/numero classes {}\n'.format(num_classes,num_visinhos,get_vizinhos))
        k = 0
        for q in range(len(teste['data'])):#range tamnho da base
            dados = dict()
            dados['data'] = list()
            cont_class=0
            vet_distancias, dist_meio=(distancia_maxima(teste, validacao, q, get_vizinhos))
            for j in validacao['data']:
                vetor2 = (j[:-1])
                for l in range(len(vet_distancias)):
                    c=euclidean4(teste['data'][q][:-1], vetor2)
                    if (c<l and cont_class<get_vizinhos and todas_as_classes[0]==j[-1]):
                        dados['data'].append(j)


                    cont_class+=1
                   # print(
                    #'Instancia teste {} , classe0 {}, classe validacao {}, cont{}'
                    #'\n'.format(q, todas_as_classes[0], j[-1], cont_class))
               # else:


            #distancias='Distancias'+nome_b+str(k)
            #cria_arff(teste, dados, distancias)
            # if num_classes==2:
            #     os.system(dcol+enderecoin+distancias+".arff"+enderecoout+str(k)+" -F 1 -N 2")
            # else:
            #     os.system(dcol + enderecoin + distancias + ".arff" + enderecoout + str(k) + " -d -F 1 -N 2")
            # k=k+1

if __name__ == '__main__':
    main()