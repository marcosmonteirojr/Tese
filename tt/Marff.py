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
    '''

    :rtype: object
    :@param dataset: arff aberto
    :@return: n_classes, classes, elementos_p_classes, total_elementos
    '''

    classes=[]#vetor com as classes
    num_class=0
    elem_p_classes=[]
    total_elementos = (len(dataset['data']))
    for i in dataset['data']:
        tmp=i[-1]
        if(tmp not in classes):
           classes.append(tmp)
           elem_p_classes.append(0)
           num_class+=1
    classes.sort()
    for i in dataset['data']:
        tmp=i[-1]
        for j in classes:
            #print(classes[j])
            if(classes[j]==tmp):
                elem_p_classes[j]=elem_p_classes[j]+1
                break

    return num_class, classes, elem_p_classes,total_elementos

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

def cria_arff(info, data, classes,pasta, nome):

        """
        cria um arquivo arff no E:
        @:param info: descricao da base/ relacao
        @:param data: dados da base
        @:param classes: classes da base
        @:param nome: do arquivo a ser gerado
        :return:
        """
        f = [str(k) for k in classes]
        info['attributes'][-1] = ('Class', f)
        obj = {
            'description': info['description'],
            'relation': info['relation'],
            'attributes': info['attributes'],
            'data': data['data'],

        }
        #print(obj)
        arq1 = arff.dumps(obj)
        arq = open(pasta+nome+'.arff','w')
        arq.write(arq1)
        arq.close()