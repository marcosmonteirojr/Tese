import sys,  Marff, os, joblib, novo_perceptron as perc
from sklearn.linear_model import Perceptron, perceptron
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.dcs.ola import OLA
from deslib.des.meta_des import METADES
from deslib.dcs.lca import LCA
from deslib.dcs.rank import Rank
from deslib.static.single_best import SingleBest
from scipy.stats import wilcoxon
from numpy import average
from numpy import std, array
from sklearn.calibration import CalibratedClassifierCV
#nome_base=sys.argv[1]

# caminho_teste = "/media/marcos/Data/Tese/Bases/Teste/"
# caminho_valida = "/media/marcos/Data/Tese/Bases/Validacao/"
# caminho = "/media/marcos/Data/Tese/AG/"
# arq=open('ResultadosFinais4.csv', 'a')
# arq1=open('AccWilcoxonD2.csv', 'a')

#print(caminho)
#nome_base='Wine'



accKUB = []
accKEB = []
accOLAB = []
accSBB = []

accKUM = []
accKEM = []
accOLAM = []
accSBM = []

accLCAB=[]
accLCAP=[]
accRankB=[]
accRankP=[]
accMetaB=[]
accMetaP=[]

def abre_arquivo(bag=None, geracao=None, valida=False, teste=False):
    global nome_base, repeticao, caminho_base, caminho_data

    #print(bag, geracao)
    if bag!=None:
        arq=open(caminho_data+str(repeticao)+"/"+nome_base+str(geracao)+".indx")
        #print(arq)
        texto = arq.readlines()
        texto=texto[bag]
       # print(texto[bag])
        #
        indx_bag=texto[:-1].split(" ")
        print((indx_bag))
        arq.close()

        indx_bag=indx_bag[1:]
        #print((indx_bag))

    elif valida:
        arq = open(caminho_base+"Validacao/" + str(repeticao) + "/" + nome_base+".idx")
        texto=arq.readline()
        indx_bag=texto.split(" ")
        arq.close()

    elif teste:
        arq = open(caminho_base+"Teste/" + str(repeticao) + "/" + nome_base+".idx")
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
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
    #print(indx_bag)
    global nome_base, classes

    #print(indx_bag)
    X_data=[]
    y_data=[]
    arq2=("/media/marcos/Data/Tese/Bases2/Dataset/"+nome_base+".arff")
    arq3=Marff.abre_arff(arq2)
    X,y,_=Marff.retorna_instacias(arq3)
    if(vet_class):
        _,classes,_,_=Marff.retorna_classes_existentes(arq3)
    for i in indx_bag:
        #print(int(i))
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    #print(X_data)
    #exit(0)
    X_data=array(X_data)
    y_data=array(y_data)
    return X_data, y_data

def cria_classificadores():
    for i in range(1,21):
       # print(i)
        global repeticao
        repeticao = i

        for j in range(1, 101):
            caminho = "/media/marcos/Data/Tese/AG/"
            bag = caminho + str(i) + "/0/Individuo" + nome_base + str(j) + '.arff'
            # bag = caminho + str(i) + "/0/Individuo" + nome_base + str(j) + '.arff'
            # moga = caminho + str(i) + "/" + str(i) + "-finais_complex/Individuo" + nome_base + str(j) + '.arff'
            #
            base_bag = Marff.abre_arff(bag)
            Xbag, ybag,_ = Marff.retorna_instacias(base_bag,True)
            model = CalibratedClassifierCV(Perceptron(max_iter=10))
            #model2 = CalibratedClassifierCV(Perceptron(), cv=3)
            # poolBag.append(percB.fit(Xbag,ybag))
            # poolPgsc.append(percP.fit(X_pgsc,y_pgsc))
            #poolBag.append(model.fit(Xbag, ybag))
            # base_moga = arff.abre_arff(moga)
            # Xmoga, ymoga = arff.retorna_instacias_numpy(base_moga)
            # percB=perceptron.Perceptron()
            # percM=perceptron.Perceptron()

            #base_bag = abre_arquivo(bag=j, geracao=0, valida=False, teste=False)
            # exit(0)
            #Xbag, ybag = monta_arquivo(base_bag)
           # print(Xbag)
            #exit(0)
            #base_pgsc = abre_arquivo(bag=j, geracao=30, valida=False, teste=False)
            #X_pgsc, y_pgsc = monta_arquivo(base_pgsc)
            #model = CalibratedClassifierCV(Perceptron(n_iter=2))
            #model2 = CalibratedClassifierCV(Perceptron(n_iter=2))
            C_bag = model.fit(Xbag, ybag)
            #C_pgsc = model2.fit(X_pgsc, y_pgsc)
            #percB = perceptron.Perceptron(n_jobs=4)
            #percP = perceptron.Perceptron(n_jobs=4)
            #C_bag = percB.fit(Xbag, ybag)
            #C_pgsc = percP.fit(X_pgsc, y_pgsc)

            #poolBag.append(C_bag)
            #poolPgsc.append(C_pgsc)

            if (os.path.exists("/media/marcos/Data/Tese/GA2/"+str(repeticao)+"/Classificadores") == False):
                os.system("mkdir -p /media/marcos/Data/Tese/GA2/"+str(repeticao)+"/Classificadores")
        #for l in range(len(poolBag)):
            joblib.dump(C_bag, "/media/marcos/Data/Tese/GA2/"+str(repeticao)+"/Classificadores/"+str(j)+nome_base+ "0.pkl")
            #joblib.dump(C_pgsc, "/media/marcos/Data/Tese/GA2/"+str(repeticao)+"/Classificadores/"+str(j)+nome_base + "30.pkl")

#print(nome_base)
def LoadClassifiers(tipo, repeticao):
    classifiers=[]
    if tipo =="bag":
        for j in range(0,100):
            classifiers.append(joblib.load("/media/marcos/Data/Tese/GA2/"+str(repeticao)+"/Classificadores/"+str(j)+nome_base+ "0.pkl"))
    if tipo =="pgsc":
        for j in range(0,100):
            classifiers.append(joblib.load("/media/marcos/Data/Tese/GA2/"+str(repeticao)+"/Classificadores/"+str(j)+nome_base+ "30.pkl"))
    return classifiers

def roda(tipo):
    global repeticao, caminho_data, caminho_base
    if (tipo == 1):
        caminho = caminho_data
        arq = open('SelecaoMedia_desvio_pgcs1.csv', 'a')
        arq1 = open('SelecaoWilcoxon_pgcs1.csv', 'a')
        arq2 = open('SelecaoTabela_pgcs1.csv', 'a')
    if(tipo==2):
        arq = open('SelecaoMedia_desvio_pgcs2.csv', 'a')
        arq1 = open('SelecaoWilcoxon_pgcs2.csv', 'a')
        arq2 = open('SelecaoTabela_pgcs2.csv', 'a')
    for i in range(1,21):
       # print(i)
        global repeticao
        repeticao = i
        poolBag = []
        poolPgsc = []
        if nome_base=='Ecoli':
            caminho_teste = "/media/marcos/Data/Tese/Bases/Teste/"
            caminho_valida = "/media/marcos/Data/Tese/Bases/Validacao/"
            t = caminho_teste + str(i) + "/Teste" + nome_base + str(i) + ".arff"
            v = caminho_valida + str(i) + "/Valida" + nome_base + str(i) + ".arff"
            base_teste = Marff.abre_arff(t)
            base_validacao = Marff.abre_arff(v)
            X_test, y_test,_ = Marff.retorna_instacias(base_teste, np_array=True)
            X_valida, y_valida,_ = Marff.retorna_instacias(base_validacao,np_array=True)
        else:
            base_teste = abre_arquivo(bag=None, geracao=None, valida=False, teste=True)
            base_validacao = abre_arquivo(bag=None, geracao=None, valida=True, teste=False)
            X_test, y_test = monta_arquivo(base_teste)
            X_valida, y_valida = monta_arquivo(base_validacao)
#################################################################################################

        if (tipo==1):
            for j in range(1, 101):
                bag = caminho + str(i) + "/0/Individuo" + nome_base + str(j) + '.arff'
                moga = caminho + str(i) + "/" + str(i) + "-finais/Individuo" + nome_base + str(j) + '.arff'

                base_bag = Marff.abre_arff(bag)
                X_bag, y_bag,_ = Marff.retorna_instacias(base_bag,True)
                base_pgsc = Marff.abre_arff(moga)
                X_pgsc, y_pgsc,_ = Marff.retorna_instacias(base_pgsc,True)

                percB = perc.PPerceptron(n_jobs=4,max_iter=10)
                percP = perc.PPerceptron(n_jobs=4,max_iter=10)

                poolBag.append(percB.fit(X_bag,y_bag))
                poolPgsc.append(percP.fit(X_pgsc,y_pgsc))

        elif (tipo==2):
            for j in range(0,100):
                base_bag = abre_arquivo(bag=j, geracao=0, valida=False, teste=False)
                X_bag, y_bag = monta_arquivo(base_bag)
                base_pgsc = abre_arquivo(bag=j, geracao=30, valida=False, teste=False)
                X_pgsc, y_pgsc = monta_arquivo(base_pgsc)

                percB = perc.PPerceptron(n_jobs=4, max_iter=10)
                percP = perc.PPerceptron(n_jobs=4,max_iter=10)

                poolBag.append(percB.fit(X_bag,y_bag))
                poolPgsc.append(percP.fit(X_pgsc,y_pgsc))


        metdb=METADES(poolBag)
        metdp=METADES(poolPgsc)
        lcab=LCA(poolBag)
        lcap=LCA(poolPgsc)
        rankb=Rank(poolBag)
        rankp=Rank(poolPgsc)

        metdb.fit(X_valida,y_valida)
        metdp.fit(X_valida,y_valida)
        lcab.fit(X_valida,y_valida)
        lcap.fit(X_valida,y_valida)
        rankb.fit(X_valida,y_valida)
        rankp.fit(X_valida,y_valida)
        #exit(0)
        accMetaB.append(metdb.score(X_test,y_test))
        accMetaP.append(metdp.score(X_test,y_test))
        accLCAB.append(lcab.score(X_test,y_test))
        accLCAP.append(lcap.score(X_test,y_test))
        accRankB.append(rankb.score(X_test,y_test))
        accRankP.append(rankp.score(X_test,y_test))
        knorauB = KNORAU(poolBag)
        kneB = KNORAE(poolBag)
        olaB = OLA(poolBag)
        singleB= SingleBest(poolBag)

        knorauM = KNORAU(poolPgsc)
        kneM = KNORAE(poolPgsc)
        olaM = OLA(poolPgsc)
        singleM = SingleBest(poolPgsc)
        #
        # #print(X_valida)
        #
        knorauB.fit(X_valida, y_valida)

        kneB.fit(X_valida, y_valida)
        olaB.fit(X_valida, y_valida)
        singleB.fit(X_valida, y_valida)
        #
        knorauM.fit(X_valida, y_valida)
        kneM.fit(X_valida, y_valida)
        olaM.fit(X_valida, y_valida)
        singleM.fit(X_valida, y_valida)
        #
        accKUB.append(knorauB.score(X_test,y_test))
        accKEB.append(kneB.score(X_test,y_test))
        accOLAB.append(olaB.score(X_test,y_test))
        accSBB.append(singleB.score(X_test,y_test))
        #
        accKUM.append(knorauM.score(X_test,y_test))
        accKEM.append(kneM.score(X_test,y_test))
        accOLAM.append(olaM.score(X_test,y_test))
        accSBM.append(singleM.score(X_test,y_test))
        #
        kp,ke=wilcoxon(accKEB,accKEM)
        kp2,ku=wilcoxon(accKUB,accKUM)
        op,ol=wilcoxon(accOLAB,accOLAM)
        sp,sb=wilcoxon(accSBB,accSBM)
        lca,lc=wilcoxon(accLCAB,accLCAP)
        rk,rkb=wilcoxon(accRankB,accRankP)
        met,mt=wilcoxon(accMetaB,accMetaP)

        #kp=ke=0
        #kp2=ku=0
        #op=ol=0
        #sp=sb=0

        print(i)

    #print(accSBB)
    arq1.write('{};{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{}\n'.format(nome_base,kp,ke,kp2,ku,op,ol,sp,sb, lca,lc, rk,rkb,met,mt))
    arq.write('{};{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{}\n'.format(nome_base,100*average(accKUB),100*std(accKUB),100*average(accKUM),
    100*std(accKUM),
    100*average(accKEB),100*std(accKEB),100*average(accKEM),100*std(accKEM),
    100*average(accOLAB),100*std(accOLAB),100*average(accOLAM),100*std(accOLAM),
    100*average(accSBB),100*std(accSBB),100*average(accSBM),100*std(accSBM),
    100*average(accLCAB),100*std(accLCAB),100*average(accLCAP), 100*std(accLCAP),
    100*average(accRankB), 100*std(accRankB),100*average(accRankP),100*std(accRankP),
    100*average(accMetaB),100*std(accMetaB),100*average(accMetaP),100*std(accMetaP)))

    arq2.write('{};{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({})\n'.format(nome_base,
    round(100*average(accKUB),1),round(100*std(accKUB),1),round(100*average(accKUM),1), round(100*std(accKUM),1),
    round(100*average(accKEB),1),round(100*std(accKEB),1),round(100*average(accKEM),1),round(100*std(accKEM),1),
    round(100*average(accOLAB),1),round(100*std(accOLAB),1),round(100*average(accOLAM),1),round(100*std(accOLAM),1),
    round(100*average(accSBB),1),round(100*std(accSBB),1),round(100*average(accSBM),1),round(100*std(accSBM),1),
    round(100*average(accLCAB),1),round(100*std(accLCAB),1), round(100*average(accLCAP),1),round(100*std(accLCAP),1),
    round(100*average(accRankB),1),round(100*std(accRankB),1),round(100*average(accRankP),1),round(100*std(accRankP),1),
    round(100*average(accMetaB),1),round(100*std(accMetaB),1),round(100*average(accMetaP),1),round(100*std(accMetaP),1)))
    arq1.close()
    arq2.close()
    arq.close()

#repeticao=1
#nome_base='Ecoli'
#caminho_data="/media/marcos/Data/Tese/GA2/"
caminho_base="/media/marcos/Data/Tese/Bases2/"
classes=[]
#roda(2)
#cria_classificadores()
#bases=open("/home/marcos/PycharmProjects/Tese/tt/Bases/bases3.txt")
# for i in bases:
#     nome_base=i
#     nome_base=nome_base[:-1].split()
#     nome_base=nome_base[0]
#     print(nome_base)
   # roda()