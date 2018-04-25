import sys,  Marff as arff
from sklearn.linear_model import perceptron
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.dcs.ola import OLA
from deslib.static.single_best import SingleBest
from scipy.stats import wilcoxon
from numpy import average
from numpy import std
nome_base=sys.argv[1]
#nome_base='Magic'
caminho_teste = "/media/marcos/Data/Tese/Bases/Teste/"
caminho_valida = "/media/marcos/Data/Tese/Bases/Validacao/"
caminho = "/media/marcos/Data/Tese/AG/"
arq=open('ResultadosFinaisSelecaoD.csv', 'a')
arq1=open('WilSelecaoD.csv', 'a')

#print(caminho)
accKUB = []
accKEB = []
accOLAB = []
accSBB = []

accKUM = []
accKEM = []
accOLAM = []
accSBM = []


print(nome_base)
for i in range(1,21):
   # print(i)


    poolBag=[]
    poolMoga = []
    t = caminho_teste + str(i) + "/Teste" + nome_base + str(i) + ".arff"
    v = caminho_valida + str(i) + "/Valida" + nome_base + str(i) + ".arff"
    base_teste = arff.abre_arff(t)
    base_validacao = arff.abre_arff(v)
    X_test, y_test = arff.retorna_instacias_numpy(base_teste)
    X_valida, y_valida = arff.retorna_instacias_numpy(base_validacao)
    for j in range(1, 101):


        bag = caminho + str(i) + "/0/Individuo" + nome_base + str(j) + '.arff'
        moga = caminho + str(i) + "/" + str(i) + "-finais/Individuo" + nome_base + str(j) + '.arff'

        base_bag = arff.abre_arff(bag)
        Xbag, ybag = arff.retorna_instacias_numpy(base_bag)

        base_moga = arff.abre_arff(moga)
        Xmoga, ymoga = arff.retorna_instacias_numpy(base_moga)
        percB=perceptron.Perceptron()
        percM=perceptron.Perceptron()

        #model_perc_Moga = CalibratedClassifierCV(perceptron(max_iter=100)).fit(Xmoga, ymoga)

        poolBag.append(percB.fit(Xbag, ybag))
        poolMoga.append(percM.fit(Xmoga,ymoga))

    knorauB = KNORAU(poolBag)
    kneB = KNORAE(poolBag)
    olaB = OLA(poolBag)
    singleB= SingleBest(poolBag)

    knorauM = KNORAU(poolMoga)
    kneM = KNORAE(poolMoga)
    olaM = OLA(poolMoga)
    singleM = SingleBest(poolMoga)

    #print(X_valida)

    knorauB.fit(X_valida, y_valida)
    #exit(0)
    kneB.fit(X_valida, y_valida)
    olaB.fit(X_valida, y_valida)
    singleB.fit(X_valida, y_valida)

    knorauM.fit(X_valida, y_valida)
    kneM.fit(X_valida, y_valida)
    olaM.fit(X_valida, y_valida)
    singleM.fit(X_valida, y_valida)

    accKUB.append(knorauB.score(X_test,y_test))
    accKEB.append(kneB.score(X_test,y_test))
    accOLAB.append(olaB.score(X_test,y_test))
    accSBB.append(singleB.score(X_test,y_test))

    accKUM.append(knorauM.score(X_test,y_test))
    accKEM.append(kneM.score(X_test,y_test))
    accOLAM.append(olaM.score(X_test,y_test))
    accSBM.append(singleM.score(X_test,y_test))

    kp,ke=wilcoxon(accKEB,accKEM)
    kp2,ku=wilcoxon(accKUB,accKUM)
    op,ol=wilcoxon(accOLAB,accOLAM)
    sp,sb=wilcoxon(accSBB,accSBM)

    print(i)

#print(accSBB)
arq1.write('{};{};{};;{};{};;{};{};;{};{}\n'.fomat(nome_base,str(kp),str(ke),str(kp2),str(ku),str(op),str(ol),str(sp),str(sb)))
arq.write('{};{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{}\n'.format(nome_base,average(accKUB),std(accKUB),average(accKUM),std(accKUM),
average(accKEB),std(accKEB),average(accKEM),std(accKEM),average(accOLAB),std(accOLAB),average(accOLAM),
std(accOLAM),average(accSBB),std(accSBB),average(accSBM),std(accSBM)))
arq1.close()
arq.close()


