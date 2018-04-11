import arff, numpy


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
    classes_d_dataset=[]
    num_class=len(dataset['attributes'][-1][1])
    elem_p_classes=[]
    total_elementos = (len(dataset['data']))

    for k in range(0,num_class):
        classes.append(k)

    for i in dataset['data']:
        tmp=i[-1]
        if(tmp not in classes_d_dataset):
           classes_d_dataset.append(tmp)
           elem_p_classes.append(0)
           #num_class+=1
    classes_d_dataset.sort()
    if(len(classes_d_dataset)!=num_class):
         print('nao extratificado')
         #print(len(classes_d_dataset))
    else:
        for i in dataset['data']:
            tmp=i[-1]
            for j in classes_d_dataset:
               # print(j)
                if(classes_d_dataset[j]==tmp):
                   elem_p_classes[j]=elem_p_classes[j]+1
                   break
    #print(num_class,classes_d_dataset, classes,elem_p_classes,total_elementos)
    return num_class,classes, classes_d_dataset, elem_p_classes,total_elementos

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

def retorna_instacias_numpy(dataset):
    """

    :param dataset: arff carregado
    :return: X e y
    """
    y=[]
    X=[]
    for i in dataset['data']:
        y.append(i[-1])
        X.append(i[:-1])

    return numpy.array(X),numpy.array(y)

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

#a=abre_arff('/media/marcos/Data/Tese/Bases/Validacao/1/ValidaWine1.arff')
#_=retorna_classes_existentes(a)