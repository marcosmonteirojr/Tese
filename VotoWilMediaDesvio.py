from scipy.stats import wilcoxon
from sklearn.linear_model import perceptron
from mlxtend.classifier import EnsembleVoteClassifier
import Marff
from numpy import average, std
import sys


nome_base=sys.argv[1]
#nome_base='Wine'
caminho_teste = "/media/marcos/Data/Tese/Bases2/Teste/"
caminho_valida = "/media/marcos/Data/Tese/Bases2/Validacao/"
caminho = "/media/marcos/Data/Tese/AG/"
arq=open('Media_desvio_voto_pgcs2.csv', 'a')
arq1=open('AccVoto_pgcs2.csv', 'a')
arq2=open('AccWilcoxon_pgcs2.csv', 'a')
arq3=open('Tabela_pgcs2.csv', 'a')
accVotingBag = []
accVotingPgsc = []

def abre_arquivo(bag=None, geracao=None, valida=False, teste=False):
    global nome_base, repeticao

    print(bag, geracao)
    if bag!=None:
        arq=open("/media/marcos/Data/Tese/GA2/"+str(repeticao)+"/"+nome_base+str(geracao)+".indx")
        #print(arq)
        texto = arq.readlines()
        texto=texto[bag]
       # print(texto[bag])
        #
        indx_bag=texto[:-1].split(" ")
        #print(len(indx_bag))
        arq.close()

        indx_bag=indx_bag[1:]
        #print((indx_bag))



    elif valida:
        arq = open("/media/marcos/Data/Tese/Bases2/Validacao/" + str(repeticao) + "/" + nome_base+".idx")
        texto=arq.readline()
        indx_bag=texto.split(" ")
        arq.close()

    elif teste:
        arq = open("/media/marcos/Data/Tese/Bases2/Teste/" + str(repeticao) + "/" + nome_base+".idx")
        texto=arq.readline()
        indx_bag=texto.split(" ")
        arq.close()
    #exit(0)
    #print(indx_bag)
    return indx_bag

def monta_arquivo(indx_bag,vet_class=False):
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :return:
    '''
    #print(indx_bag)
    global nome_base, classes

    #print(indx_bag)
    X_data=[]
    y_data=[]
    arq2=("/media/marcos/Data/Tese/Bases2/Dataset/"+nome_base+".arff")
    arq3=Marff.abre_arff(arq2)
    X,y=Marff.retorna_instacias(arq3)
    if(vet_class):
        _,classes,_,_=Marff.retorna_classes_existentes(arq3)
    for i in indx_bag:
        #print(int(i))
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    #print(X_data)
    #exit(0)
    return X_data, y_data



for i in range(1,21):
   # print(i)
    repeticao=i


    poolBag=[]
    poolPgsc = []

    base_teste = abre_arquivo(bag=None,geracao=None,valida=False,teste=True)
    base_validacao = abre_arquivo(bag=None,geracao=None,valida=True,teste=False)
    X_test, y_test = monta_arquivo(base_teste)
    X_valida, y_valida = monta_arquivo(base_validacao)
    for j in range(0, 100):




        base_bag = abre_arquivo(bag=j,geracao=0,valida=False,teste=False)
        #exit(0)
        Xbag, ybag = monta_arquivo(base_bag)
        base_pgsc = abre_arquivo(bag=j,geracao=30,valida=False,teste=False)
        X_pgsc, y_pgsc = monta_arquivo(base_pgsc)
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

arq.write('{};{};{};;{};{}\n'.format(nome_base,average(accVotingBag),std(accVotingBag),average(accVotingPgsc),std(accVotingPgsc)))
arq3.write('{};{} ({});;{} ({})\n'.format(nome_base,average(accVotingBag),std(accVotingBag),average(accVotingPgsc),std(accVotingPgsc)))
arq2.write('{};{};{}\n'.format(nome_base,p,w))
arq.close()
arq1.close()
arq2.close()
arq3.close()
