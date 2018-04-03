import sys, newDcol, Marff as arff
from sklearn.linear_model import perceptron

from numpy import average
#nome_base=sys.argv[1]
nome_base='WDVG'
caminho_teste = "/media/marcos/Data/Tese/Bases/Teste/"
caminho = "/media/marcos/Data/Tese/AG/"
arq=open('./tt/ResultadosFinais.csv', 'a')
arq.write('Nome_base;Repeticao;AccBag;AccPGSC\n')
#print(caminho)


for i in range(1,21):
   # print(i)
    accBag = []
    accMoga = []
    v = caminho_teste + str(i) + "/Teste" + nome_base + str(i) + ".arff"
    base_teste = arff.abre_arff(v)
    X_test, y_test = arff.retorna_instacias(base_teste)
    for j in range(1, 101):
        bag = caminho + str(i) + "/0/Individuo" + nome_base + str(j) + '.arff'
        moga = caminho + str(i) + "/" + str(i) + "-finais/Individuo" + nome_base + str(j) + '.arff'
        base_bag = arff.abre_arff(bag)
        Xbag, ybag = arff.retorna_instacias(base_bag)
        base_moga = arff.abre_arff(moga)
        Xmoga, ymoga = arff.retorna_instacias(base_moga)

        perc_bag = perceptron.Perceptron()
        perc_bag.fit(Xbag, ybag)
        accBag.append(perc_bag.score(X_test, y_test))

        perc_moga = perceptron.Perceptron()
        perc_moga.fit(Xmoga, ymoga)
        accMoga.append(perc_moga.score(X_test, y_test))
    arq.write('{},{},{},{}\n'.format(nome_base,str(i),average(accBag),average(accMoga)))
    print('BAg',average(accBag))
    print(nome_base, i)
arq.close()


