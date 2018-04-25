from scipy.stats import wilcoxon
from sklearn.linear_model import perceptron
from mlxtend.classifier import EnsembleVoteClassifier
import Marff as arff
from numpy import average, std
import sys


#nome_base=sys.argv[1]
nome_base='Wine'
caminho_teste = "/media/marcos/Data/Tese/Bases/Teste/"
caminho_valida = "/media/marcos/Data/Tese/Bases/Validacao/"
caminho = "/media/marcos/Data/Tese/AG/"
arq=open('Media_desvio2.csv', 'a')
arq1=open('AccVoto2.csv', 'a')
arq2=open('AccWilcoxon2.csv', 'a')
accVotingBag = []
accVotingPgsc = []


for i in range(1,21):
   # print(i)


    poolBag=[]
    poolPgsc = []
    t = caminho_teste + str(i) + "/Teste" + nome_base + str(i) + ".arff"
    v = caminho_valida + str(i) + "/Valida" + nome_base + str(i) + ".arff"
    base_teste = arff.abre_arff(t)
    base_validacao = arff.abre_arff(v)
    X_test, y_test = arff.retorna_instacias_numpy(base_teste)
    X_valida, y_valida = arff.retorna_instacias_numpy(base_validacao)
    for j in range(1, 101):


        bag = caminho + str(i) + "/0/Individuo" + nome_base + str(j) + '.arff'
        pgsc = caminho + str(i) + "/" + str(i) + "-finais_complex/Individuo" + nome_base + str(j) + '.arff'

        base_bag = arff.abre_arff(bag)
        Xbag, ybag = arff.retorna_instacias_numpy(base_bag)
        base_pgsc = arff.abre_arff(pgsc)
        X_pgsc, y_pgsc = arff.retorna_instacias_numpy(base_pgsc)
        percB = perceptron.Perceptron()
        percP = perceptron.Perceptron()

        poolBag.append(percB.fit(Xbag, ybag))
        poolPgsc.append(percP.fit(X_pgsc, y_pgsc))


    #exit(0)
    bagging_voting=EnsembleVoteClassifier(clfs=poolBag,voting='hard',refit=False)
    bagging_vot=bagging_voting.fit(X_valida,y_valida)

    B=100*bagging_vot.score(X_test,y_test)
    Pgsc_voting = EnsembleVoteClassifier(clfs=poolPgsc, voting='hard', refit=False)
    Pgsc_voting=Pgsc_voting.fit(X_valida, y_valida)

    P=100*Pgsc_voting.score(X_test,y_test)
    accVotingBag.append(B)
    accVotingPgsc.append(P)

   # print(P)
    #print(accVotingPgsc)
    #exit(0)
    arq1.write('{};{};{}\n'.format(nome_base,B,P))
arq1.write('\n')

p,w=wilcoxon(accVotingBag,accVotingPgsc)

arq.write('{};{};{};;{};{}\n'.format(nome_base,average(accVotingBag),std(accVotingBag),average(accVotingPgsc),std(accVotingBag)))
arq2.write('{};{};{}\n'.format(nome_base,p,w))
arq.close()
arq1.close()
arq2.close()
