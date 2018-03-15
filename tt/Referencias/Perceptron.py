import arff, os
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import perceptron

nome_base="Wine"
lables=[]#variavel com lables das classes
dados = list()#variaveis com demais atributos das classes
treino = arff.load(open('/home/marcos/Documents/Tese/Bases/Treino/1/Treino'+nome_base+'1.arff'))
teste = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))
nome_base="Wine"
resultados=[]
#net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True)
print(treino['data'])
if (os.path.exists("clf_pec"+nome_base+".pkl") == False):#verifica se exite um job do classificador
    perc = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True)#chama o indutor perceptron
    for i in range(len(treino['data'])):#percorre a base treino e separa os lables das classes
        lables.append(treino['data'][i][13])
    for i in treino['data']:#percorre a base treino e salva os atributos sem a classe
        dados.append(i[:-1])

    perc.fit(dados,lables)#treina o perceptron
    #print(perc.score(dados,lables))
    joblib.dump(perc, "clf_pec"+nome_base+".pkl")#salva o job
else:
    perc = joblib.load("clf_pec"+nome_base+".pkl")# se ja existe o job carrega (job= classificador completo)
    for i in teste["data"]:#tira o atributo classe
        dados.append(i[:-1])
    for i in range (len(teste['data'])):
         dados2=np.array([dados[i]])#transforma para o formato do predict
         resultados.append(perc.predict(dados2))#testa o classificador treinado, carregado do job
resultados=np.array(resultados)
for i in resultados:
    if i == "1":
        print('e')
    elif i == "2":
        print('n')
    #elif i == "3":

