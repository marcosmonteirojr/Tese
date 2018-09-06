import novo_perceptron as perc
import sys,  Marff, random
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.dcs.ola import OLA
from deslib.des.meta_des import METADES
from deslib.dcs.lca import LCA
from deslib.dcs.rank import Rank
from deslib.static.single_best import SingleBest
from scipy.stats import wilcoxon
from numpy import average,std, array, argsort
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import complexity_pcol as dcol
from mlxtend.classifier import EnsembleVoteClassifier


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

accVotingBag=[]
accVotingPgsc=[]

def abre_arquivo(bag=None, geracao=None, valida=False, teste=False, experimento=None):
    global nome_base, repeticao, caminho_base, caminho_data

    #print(bag, geracao)
    if bag!=None:
        if experimento:
            arq=open(caminho_data+str(repeticao)+"/"+nome_base+str(geracao)+experimento+".indx")
        else:
            arq = open(caminho_data + str(repeticao) + "/" + nome_base + str(geracao) + ".indx")
        #print(arq)
        texto = arq.readlines()
        texto=texto[bag]
       # print(texto[bag])
        #
        indx_bag=texto[:-1].split(" ")
       # print((indx_bag))
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
def chunks(lista, n):
    inicio = 0
    for i in range(n):
        final = inicio + len(lista[i::n])
        yield lista[inicio:final]
        inicio = final
    return lista

def monta_arquivo(indx_bag,vet_class=False):
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
    #print(indx_bag)
    global nome_base, classes, caminho_base

    #print(indx_bag)
    X_data=[]
    y_data=[]
    arq2=(caminho_base+"/Dataset/"+nome_base+".arff")
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


def retorna_lista_maiores_distancias(arq, nc, X_val, y_val):
    global classes
    compx = []
    dist = []
    nome_bag = []
    teste=[]
    texto = arq.readlines()
    pool=[]
    pool_score=[]
    pool_final = []
    pool_accuracia = []
    #print((texto), '3-arquivo 29')
    for i in range(len(texto)):
        # if (i <= 99):

        text = texto[i]
        indx_bag = text.split(" ")
        nome_bag.append(indx_bag[0])
        indx_bag = indx_bag[1:]

        X_29, y_29 = monta_arquivo(indx_bag, True)
        scaler = MinMaxScaler()
        scaler.fit(X_29)
        transformed_data = scaler.transform(X_29)

        complex = dcol.PPcol(classes=classes)
        complexidades = complex.xy_measures(transformed_data, y_29)
        F1 = (average(complexidades['F1']))
        N2 = (average(complexidades['N2']))
        cpx = [F1, N2]
        compx.append(cpx)
        per = perc.PPerceptron(n_jobs=4, max_iter=10)
        pool.append(per.fit(X_29, y_29))
        pool_score.append(per.score(X_val,y_val))


    for k in range(len(compx)):
        dista = 0
        for l in range(len(compx)):
            if (k == l):
                continue
            else:
                dista += distance.euclidean(compx[k], compx[l])
        dist.append(dista / len(texto))
    f = argsort(dist)

    if nc==2:
        while len(teste)!=20:
            x=f[:120]
            teste.append(random.choice(x) )
        while len(teste) != 40:
            x = f[120:240]
            teste.append(random.choice(x) )
        while len(teste) != 60:
            x = f[240:360]
            teste.append(random.choice(x) )
        while len(teste) != 80:
            x = f[360:480]
            teste.append(random.choice(x) )
        while len(teste) != 100:
            x = f[480:600]
            teste.append(random.choice(x) )
        for i in teste:
            pool_final.append(pool[i])
    if nc==1:
        for i in f[::-1]:
            if len(pool_final) < 100:
                pool_final.append(pool[i])
    if nc==3:

        x=list(chunks(f,100))
        for i in range(len(x)):
            maior = 0
            for j in range(len(x[i])):
                aux=pool_score[x[i][j]]
                print(aux)
                if maior < aux:
                   maior=aux
                   y=x[i][j]
            pool_accuracia.append(y)
        for i in pool_accuracia:
            pool_final.append(pool[i])
    del f, texto, compx, dist,teste, pool_accuracia,pool

    return nome_bag, pool_final

def roda(tipo,tipo2=None):
    '''

    :param tipo: 1 para arff, 2 indices (100 individuos), 3 para 600 individuos, podendo ser Proximo acurva da dispersao ou dividindo o espaco
    :param nc: True->proximo a curva da dispersao,, False->divide os espaco e seleciona alguns bags
    :return:
    '''
    global repeticao, caminho_data, caminho_base
    if (tipo == 1):
        caminho = caminho_data
        arq = open('SelecaoMedia_desvio_pgcs1.csv', 'a')
        arq1 = open('SelecaoWilcoxon_pgcs1.csv', 'a')
        arq2 = open('SelecaoTabela_pgcs1.csv', 'a')
    if(tipo==2 or tipo==3):
        arq = open('SelecaoMedia_desvio_pgcs3.csv', 'a')
        arq1 = open('SelecaoWilcoxon_pgcs3.csv', 'a')
        arq2 = open('SelecaoTabela_pgcs3.csv', 'a')
    if (tipo == 4):
        arq = open('SelecaoMedia_desvio_pgcs4.csv', 'a')
        arq1 = open('SelecaoWilcoxon_pgcs4.csv', 'a')
        arq2 = open('SelecaoTabela_pgcs4.csv', 'a')
    for i in range(1,2):
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
                #print(len(poolPgsc))

        elif (tipo==2):
            for j in range(0,100):
                base_bag = abre_arquivo(bag=j, geracao=0, valida=False, teste=False)
                X_bag, y_bag = monta_arquivo(base_bag)
                base_pgsc = abre_arquivo(bag=j, geracao=30, valida=False, teste=False, experimento="-3")
                X_pgsc, y_pgsc = monta_arquivo(base_pgsc)

                percB = perc.PPerceptron(n_jobs=4, max_iter=10)
                percP = perc.PPerceptron(n_jobs=4,max_iter=10)

                poolBag.append(percB.fit(X_bag,y_bag))
                poolPgsc.append(percP.fit(X_pgsc,y_pgsc))
        elif (tipo==3):
            arq_bags = open(caminho_bags + str(repeticao) + "/" + nome_base + "29.indx")
            _,poolPgsc = retorna_lista_maiores_distancias(arq_bags,tipo2,X_valida,y_valida)
            for j in range(0, 100):
                base_bag = abre_arquivo(bag=j, geracao=0, valida=False, teste=False)
                X_bag, y_bag = monta_arquivo(base_bag)
                percB = perc.PPerceptron(n_jobs=4, max_iter=10)
                poolBag.append(percB.fit(X_bag, y_bag))
           # print(nome_bag)
        elif tipo==4:
            arq_bags = open(caminho_bags + str(repeticao) + "/" + nome_base + "29.indx")
            # print(caminho_bags + str(repeticao) + "/" + nome_base + "29.indx")
            _,poolPgsc = retorna_lista_maiores_distancias(arq_bags, 3, X_val=X_valida,y_val=y_valida)
            #print(len(poolPgsc))
            for j in range(0, 100):
                base_bag = abre_arquivo(bag=j, geracao=0, valida=False, teste=False)
                X_bag, y_bag = monta_arquivo(base_bag)
                percB = perc.PPerceptron(n_jobs=4, max_iter=10)
                poolBag.append(percB.fit(X_bag, y_bag))


        #exit(0)
        bagging_voting = EnsembleVoteClassifier(clfs=poolBag, voting='hard', refit=False)
        bagging_vot = bagging_voting.fit(X_valida, y_valida)
        B =  bagging_vot.score(X_test, y_test)


        #teste=[]
        Pgsc_voting = EnsembleVoteClassifier(clfs=poolPgsc, voting='hard', refit=False)
        Pgsc_voting = Pgsc_voting.fit(X_valida, y_valida)
        P =  Pgsc_voting.score(X_test, y_test)
        knorauB = KNORAU(poolPgsc)#########33
        knorauB.fit(X_valida, y_valida)
        knorauB.score(X_test,y_test)
        #for i in range (1,2):
        teste = []
        teste.append(X_valida[0])
        print(len(knorauB.select(teste[0])))

    #     metdb=METADES(poolBag)
    #     metdp=METADES(poolPgsc)
    #     lcab=LCA(poolBag)
    #     lcap=LCA(poolPgsc)
    #     rankb=Rank(poolBag)
    #     rankp=Rank(poolPgsc)
    #
    #     metdb.fit(X_valida,y_valida)
    #     metdp.fit(X_valida,y_valida)
    #     lcab.fit(X_valida,y_valida)
    #     lcap.fit(X_valida,y_valida)
    #     rankb.fit(X_valida,y_valida)
    #     rankp.fit(X_valida,y_valida)
    #     #exit(0)
    #     accMetaB.append(metdb.score(X_test,y_test))
    #     accMetaP.append(metdp.score(X_test,y_test))
    #     accLCAB.append(lcab.score(X_test,y_test))
    #     accLCAP.append(lcap.score(X_test,y_test))
    #     accRankB.append(rankb.score(X_test,y_test))
    #     accRankP.append(rankp.score(X_test,y_test))
    #     knorauB = KNORAU(poolBag)
    #     kneB = KNORAE(poolBag)
    #     olaB = OLA(poolBag)
    #     singleB= SingleBest(poolBag)
    #
    #     knorauM = KNORAU(poolPgsc)
    #     kneM = KNORAE(poolPgsc)
    #     olaM = OLA(poolPgsc)
    #     singleM = SingleBest(poolPgsc)
    #     #
    #     knorauB.fit(X_valida, y_valida)
    #     kneB.fit(X_valida, y_valida)
    #     olaB.fit(X_valida, y_valida)
    #     singleB.fit(X_valida, y_valida)
    #     #
    #     knorauM.fit(X_valida, y_valida)
    #     kneM.fit(X_valida, y_valida)
    #     olaM.fit(X_valida, y_valida)
    #     singleM.fit(X_valida, y_valida)
    #     #
    #     accKUB.append(knorauB.score(X_test,y_test))
    #     accKEB.append(kneB.score(X_test,y_test))
    #     accOLAB.append(olaB.score(X_test,y_test))
    #     accSBB.append(singleB.score(X_test,y_test))
    #     accVotingBag.append(B)
    #     #
    #     accKUM.append(knorauM.score(X_test,y_test))
    #     accKEM.append(kneM.score(X_test,y_test))
    #     accOLAM.append(olaM.score(X_test,y_test))
    #     accSBM.append(singleM.score(X_test,y_test))
    #     accVotingPgsc.append(P)
    #     #
    #     kp,ke=wilcoxon(accKEB,accKEM)
    #     kp2,ku=wilcoxon(accKUB,accKUM)
    #     op,ol=wilcoxon(accOLAB,accOLAM)
    #     sp,sb=wilcoxon(accSBB,accSBM)
    #     lca,lc=wilcoxon(accLCAB,accLCAP)
    #     rk,rkb=wilcoxon(accRankB,accRankP)
    #     met,mt=wilcoxon(accMetaB,accMetaP)
    #     vot,votc=wilcoxon(accVotingBag,accVotingPgsc)
    #
    #
    #
    #
    # print(i)
    #
    # #print(accSBB)
    # arq1.write('{};{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{}\n'.format(nome_base,kp,ke,kp2,ku,op,ol,sp,sb, lca,lc, rk,rkb,met,mt,vot,votc))
    # arq.write('{};{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{}\n'.format(nome_base,100*average(accKUB),100*std(accKUB),100*average(accKUM),
    # 100*std(accKUM),
    # 100*average(accKEB),100*std(accKEB),100*average(accKEM),100*std(accKEM),
    # 100*average(accOLAB),100*std(accOLAB),100*average(accOLAM),100*std(accOLAM),
    # 100*average(accSBB),100*std(accSBB),100*average(accSBM),100*std(accSBM),
    # 100*average(accLCAB),100*std(accLCAB),100*average(accLCAP), 100*std(accLCAP),
    # 100*average(accRankB), 100*std(accRankB),100*average(accRankP),100*std(accRankP),
    # 100*average(accMetaB),100*std(accMetaB),100*average(accMetaP),100*std(accMetaP),
    # 100*average(accVotingBag),100*std(accVotingBag),100*average(accVotingPgsc),100*std(accVotingPgsc)))
    #
    # arq2.write('{};{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({})\n'.format(nome_base,
    # round(100*average(accKUB),1),round(100*std(accKUB),1),round(100*average(accKUM),1), round(100*std(accKUM),1),
    # round(100*average(accKEB),1),round(100*std(accKEB),1),round(100*average(accKEM),1),round(100*std(accKEM),1),
    # round(100*average(accOLAB),1),round(100*std(accOLAB),1),round(100*average(accOLAM),1),round(100*std(accOLAM),1),
    # round(100*average(accSBB),1),round(100*std(accSBB),1),round(100*average(accSBM),1),round(100*std(accSBM),1),
    # round(100*average(accLCAB),1),round(100*std(accLCAB),1), round(100*average(accLCAP),1),round(100*std(accLCAP),1),
    # round(100*average(accRankB),1),round(100*std(accRankB),1),round(100*average(accRankP),1),round(100*std(accRankP),1),
    # round(100*average(accMetaB),1),round(100*std(accMetaB),1),round(100*average(accMetaP),1),round(100*std(accMetaP),1),
    #                                                                                                                                                                 round(100*average(accVotingBag),1),round(100*std(accVotingBag),1),round(100*average(accVotingPgsc),1),round(100*std(accVotingPgsc),1)))
    # arq1.close()
    # arq2.close()
    # arq.close()

#repeticao=1
nome_base='Banana'
caminho_data="/media/marcos/Data/Tese/AG/"
caminho_base="/media/marcos/Data/Tese/Bases2/"
caminho_bags="/media/marcos/Data/Tese/AG/"
#caminho_data = "/home/monteiro/Marcos/GA3/"
#caminho_base = "/home/monteiro/Marcos/Bases2/"
#caminho_bags = "/home/monteiro/Marcos/GA3/"
classes=[]
roda(1)


