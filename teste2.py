import Cpx, Marff,os,sys, random, numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
#nome_base="Banana"
local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
nome_base='Adult'


np.set_printoptions(threshold=np.nan)
X,y,*_=Cpx.open_data(nome_base,local_dataset)


bag=Cpx.open_bag(cpx_caminho+str(1)+'/',nome_base+'')

indxTest,indxVal=Cpx.open_test_vali(local,nome_base,str(1))
indxTrain=Cpx.open_training(local,nome_base,str(1))
X_val,y_val=Cpx.biuld_x_y(indxVal,X,y)
X_test,y_test=Cpx.biuld_x_y(indxTest,X,y)
X_train,y_train=Cpx.biuld_x_y(indxTrain,X,y)

def create_pool(bag, X,y,X_val,y_val):
    '''
    retorna o pool de classificadores
    :param bag: indices de instancias do bag
    :param X: valores de X do Dataset
    :param y: valores de y do Dataset
    :param X_val: validacao ou teste(usados na Cpx.biuld_classifier
    :param y_val: idem ao X_val
    :return: o pool
    '''
    pool = []
    for i in range(len(bag['nome'])):
        X_bag,y_bag=Cpx.biuld_x_y(bag['inst'][i],X,y)

        perc,*_=Cpx.biuld_classifier(X_bag,y_bag,X_val,y_val)
        pool.append(perc)
    return pool

def create_pool2(bag_x,bag_y,X_val,y_val,pool):

    perc, *_ = Cpx.biuld_classifier(bag_x, bag_y, X_val, y_val)
    pool.append(perc)

    return pool

def list_acert(pool, X_test, y_test):
    '''
    lista_acertos [classificador[instancias acertadas 1 ou nao 0]],lista com a soma de acerto por instanca (valores == 100 todos cla
    sificadores acertaram aquela instancia)
    :param pool: pool de classificadores
    :param X_test: X_teste
    :param y_test: y_teste
    :return: lista com os acertos [classificador][acertos ou erros], lista com a soma de acerto por instanca
    '''
    lista_acertos = np.zeros((100,len(y_test)), dtype=int)

    classificador_1 = []
    for j in range(len(pool)):
        x=pool[j].predict(X_test)
        for i in range(len(x)):
            if x[i]==y_test[i]:
                lista_acertos[j][i]=1
    lista_soma_acerto=np.sum(lista_acertos, axis=0)
    #print(lista_soma_acerto)
     #classificadores onde a instancia acertada somente 1 vez
    indice_inst_1, =np.where(lista_soma_acerto==1)#instancias acertadas somente 1 vez
    indice_inst_0, =np.where(lista_soma_acerto==0)

    if (indice_inst_1!=[]):
        for i in indice_inst_1:
            for j in range(len(lista_acertos)):
                if lista_acertos[j][i]==1:
                    classificador_1.append(j)
                    break
    else:
        print('N ha 1 acerto')

    return lista_acertos, indice_inst_0, indice_inst_1, classificador_1

def Oracle_Nivel1(lista_acertos):
    '''
    retorna a quantidade de instancias acertadas e o numero de acertos [soma da quantidade de um determinado numero de acertos]
    [soma de antas vezes ele aparece]
    :param lista_acertos:
    :return: unique elementos é uma lista com os valores dos acertos ex: 0 acertos 1, acerto, 20 acerto, etc. counts_elementos(quantidade de de 0 acertos, etc), por fim retorna o
    Oracle quantidade de 0 acertos/total de instancias de teste. (counts_elements[0] / len(y_test)
    '''
    oracle=np.zeros((len(y_test),),dtype=int)
    for i in range(len(lista_acertos)):
        for j in range(len(lista_acertos[i])):
            if lista_acertos[i][j]==1:
                oracle[j]=oracle[j]+1
    unique_elements, counts_elements = np.unique(oracle, return_counts=True)
    if unique_elements[0]==0:
        Oracle_final = counts_elements[0] / len(y_test)
    else:
        Oracle_final=1

    return unique_elements, counts_elements, Oracle_final

def inst_false(lista_acertos):
    '''
    instancias_erradas todas as instancias 0 a n, do teste que foram classifcadas por cada classificador
    [classificador[instancias (indices) da lista acerto]]
    :param lista_acertos: lista de acertos [classificador][acertos ou erros 1,0]
    :return: indice das instancias erradas [classificador][indices]
    '''
    instancias_erradas=[]
    for i in lista_acertos:
        erros,  = np.where(i == 0)
        instancias_erradas.append(erros.tolist())


    return instancias_erradas

def del_pool(lista_acertos,indice_inst_0, indice_inst_1, pool):
    '''
    retira os piores classificadores (os que tem pior resultado na classificação inclusive os que acertam somente uma instancia)
    :param lista_acertos:
    :param indice_inst_0: instancias nao classificadas
    :param pool: pool original
    :return: pool menor
    '''
    su=[]
    for i in lista_acertos:
        su.append(sum(i))
    sort=np.argsort(su)#piores classificadores
    cont=len(indice_inst_0)
    cont=cont+len(indice_inst_1)#retirar se quiser manter os classificadores que acertam 1 instancia
    #print('instancias 0', indice_inst_0)
    #print('instancias 1',indice_inst_1)
    #print('classificadores_1', classificador_1)
    #print('delete classificadores')
    delete_classificadores=sort[:cont]
    delete_classificadores=delete_classificadores.tolist()
    #print(delete_classificadores)

    #for i in delete_classificadores:manster os classificadores que acertam 1 instancia
    #    if i in classificador_1:
     #       delete_classificadores.remove(i)
    delete_classificadores=np.sort(delete_classificadores)
    #print(delete_classificadores)
    pool1=[]
    for i in range(len(pool)):
        if i not in delete_classificadores:

            pool1.append(pool[i])
    del pool

    return pool1


def vizinhaca_val(indice_inst_0, indice_inst_1, k, X_train, y_train, X_val, y_val):
    scaler = StandardScaler()

    X_train1 = scaler.fit_transform(X_train)
    X_train1 = scaler.transform(X_train1)
    X_val1 =scaler.transform(X_val)
    tam=np.around(len(X_train)/2)

    bags_x=[]
    bags_y=[]

    for i in indice_inst_0:
        dist=[]
        val=X_val1[i]
        for j in X_train1:
            dist.append(distance.euclidean(val,j))#distancias dos vizinhos mais proximos das instancias nao classiifcadas
        sort=np.argsort(dist)
        cont=0
        l=0

        while cont< k:
            bags_x.append(X_train[sort[l]])#criando o bag com o tamanho de k vizinhos
            bags_y.append(y_train[sort[l]])
            l=l+1
            cont=cont+1
   # for i in indice_inst_1: #acrescentando as instancias acertadas 1 vez
     #   bags_x.append(X_val[i])
      #  bags_y.append(y_val[i])

    tam=tam-len(bags_x)
    cont = 0
    while cont!=tam: #completando o bag com instancias aleatorios do treino
        d=len(X_train)
        r=random.randint(0,d-1)
        bags_x.append(X_train[r])
        bags_y.append(y_train[r])
        cont=cont+1
    #print(len(bags_y))
    return bags_x, bags_y


pool=create_pool(bag,X,y,X_val,y_val)
lista_acertos,indice_inst_0, indice_inst_1, classificador_1 =list_acert(pool,X_val,y_val)
#instancias_erradas=inst_false(lista_acertos)
acertos,soma, oracle=Oracle_Nivel1(lista_acertos)

pool=del_pool(lista_acertos,indice_inst_0, indice_inst_1, pool)
f=0
while acertos[0]!=2 or f!=200:
    print(acertos[0],soma[0])
    pool = del_pool(lista_acertos, indice_inst_0, indice_inst_1, pool)
    while len(pool)!=100:
        bags_x, bags_y = vizinhaca_val(indice_inst_0, indice_inst_1, 10, X_train, y_train, X_val, y_val)
        pool=create_pool2(bags_x,bags_y,X_val,y_val,pool)
    f=f+1
    lista_acertos, indice_inst_0, indice_inst_1, classificador_1 = list_acert(pool, X_val, y_val)
    acertos, soma, oracle = Oracle_Nivel1(lista_acertos)

    print(f)



#instancias_erradas=inst_false(lista_acertos)
acertos,soma, oracle=Oracle_Nivel1(lista_acertos)
print(acertos,soma, oracle)
#print(np.sum(lista_acertos, axis=0))

#X=np.array(X)
#y=np.array(y)
#X_test=np.array(X_test)
#y_test=np.array(y_test)
#print(counts_elements[0]/len(y_test))

#print(Cpx.oracle(pool,X,y,X_test,y_test))

