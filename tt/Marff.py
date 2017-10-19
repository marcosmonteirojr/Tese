import arff


def abre_arff(caminho):
    """

    :param caminho: caminho arff
    :return: base
    """
    base=arff.load(open(caminho),encode_nominal=True)
    #print(base)
    return base

def retorna_classes_existentes(dataset):
    """
    :param dataset: pass
    :return:
    """
    classes=[]#vetor com as classes
    num_class=0
    for i in dataset['data']:
        tmp=i[-1]
        if(tmp not in classes):
           classes.append(tmp)
           num_class+=1

    return num_class, classes

def retorna_instacias(dataset):
    """

    :param dataset: arff carregado
    :return: X e y
    """
    y=[]
    X=[]
    for i in dataset['data']:
        y.append(i[-1])
        X.append(i[:-1])
    return X,y

def cria_arff(dataset, instancias, nome, pasta):
    """
    cria um arquivo arff no E:
    @:param dataset: descricao da base/ relacao
    @:param instancias: instancias
    @:param nome: do arquivo a ser gerado
    @:param pasta: local para salvar
    :return:
    """
    obj = {
        'description': dataset['description'],
        'relation': dataset['relation'],
        'attributes': dataset['attributes'],
        'data': instancias['data'],

    }
    arq1=arff.dumps(obj)
    arq=open(pasta+nome+'.arff','w')
    arq.write(arq1)
    arq.close()