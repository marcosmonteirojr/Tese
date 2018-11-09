
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

def retorna_vet_dist_pool(arq,newga=2, dis=True, ord=True,pool_final=True, pool_sc=True,X_val=None,y_val=None):
    global classes
    '''
    :param arq: arquivo aberto pelo open
    :param ord: ordena por indices do maior para o menor
    :param pool_final: retorna o pool
    :param pool_sc: pool treinado com score, sobre a validacao
    :param X_val: x validacao
    :param y_val: y validacao
    :return: distancia, nome dos bags,pool, score
    '''

    texto = arq.readlines()
    pool_score=[]
    nome_bag=[]
    compx=[]
    dist=[]
    pool=[]
    for i in range(len(texto)):
        text = texto[i]
        if newga==2:
            indx_bag = text[:-1].split(" ")
            nome_bag.append(indx_bag[0])
            indx_bag = indx_bag[1:]
            print(indx_bag)
            exit(0)
        elif newga==3:
            indx_bag = text.split(" ")
            nome_bag.append(indx_bag[0])
            indx_bag=indx_bag[1:-1]
            print(indx_bag)
            exit(0)
        X_29, y_29 = monta_arquivo(indx_bag, True)
        scaler = MinMaxScaler()
        if pool_final:
            per = perc.PPerceptron(n_jobs=4, max_iter=100)
            pool.append(per.fit(X_29, y_29))
        if pool_sc:
            pool_score.append(per.score(X_val,y_val))
        if dis:
            scaler.fit(X_29)
            transformed_data = scaler.transform(X_29)
            complex = dcol.PPcol(classes=classes)
            complexidades = complex.xy_measures(transformed_data, y_29)
            F1 = (average(complexidades['F1']))
            N2 = (average(complexidades['N2']))
            cpx = [F1, N2]
            compx.append(cpx)
    if dis:
        for k in range(len(compx)):
            dista = 0
            for l in range(len(compx)):
                if (k == l):
                    continue
                else:
                    dista += distance.euclidean(compx[k], compx[l])
            dist.append(dista / len(texto))
        f = argsort(dist)
        del texto,compx
        if ord:
            return f, nome_bag, pool, pool_score
        else:
            return dist, nome_bag, pool, pool_score
    else:
        return nome_bag, pool, pool_score

def divide_pool_espaco(pool_ord,pool,tamanho):
    temp=[]
    pool_final=[]
    div=len(pool_ord)/tamanho

    div=int(div)

    #print(len(pool),div,tamanho)
    #exit(0)
    for i in range(1,100,div):
        print(i, tamanho-1, div)
        while len(temp) != div:
            x = pool_ord[i:div]
            temp.append(random.choice(x))
        div=div+tamanho+10
       # print(i, tamanho, div)
    #print(temp, len(temp))
    #exit(0)
    for i in temp:
        pool_final.append(pool[i])
    del temp
    #print(pool_final)
    return pool_final


def selecao(arquivo,ga):

    arq = open('SelecaoMedia_desvio_pgcs' + arquivo + '.csv', 'a')
    arq1 = open('SelecaoWilcoxon_pgcs' + arquivo + '.csv', 'a')
    arq2 = open('SelecaoTabela_pgcs' + arquivo + '.csv', 'a')
    accKUB = []
    accKEB = []
    accOLAB = []
    accSBB = []

    accKUP = []
    accKEP = []
    accOLAP = []
    accSBP = []

    accLCAB = []
    accLCAP = []
    accRankB = []
    accRankP = []
    accMetaB = []
    accMetaP = []

    accVotingBag = []
    accVotingPgsc = []
    for i in range(1,21):
        repeticao=i
        base_teste = abre_arquivo(bag=None, geracao=None, valida=False, teste=True)
        base_validacao = abre_arquivo(bag=None, geracao=None, valida=True, teste=False)
        X_test, y_test = monta_arquivo(base_teste)
        X_valida, y_valida = monta_arquivo(base_validacao)

        arq_bags = open(caminho_bags + str(repeticao) + "/" + nome_base + "0.indx")
        _,poolB,_ = retorna_vet_dist_pool(arq_bags,newga=2,dis=False,pool_final=True, pool_sc=False)
        #print(poolB,len(poolB))

        arq_P = open(caminho_data + str(repeticao) + "/" + nome_base + "20-"+arquivo+".indx")
        nome_bag, poolP, _ = retorna_vet_dist_pool(arq_P, newga=ga,dis=False, pool_final=True, pool_sc=False)
        #poolP= divide_pool_espaco(f, poolPP, 10)
        #exit(0)
        print(len(poolP))
        bagging_voting = EnsembleVoteClassifier(clfs=poolB, voting='hard', refit=False)
        bagging_vot = bagging_voting.fit(X_valida, y_valida)
        B = bagging_vot.score(X_test, y_test)

        # teste=[]
        Pgsc_voting = EnsembleVoteClassifier(clfs=poolP, voting='hard', refit=False)
        Pgsc_voting = Pgsc_voting.fit(X_valida, y_valida)
        P = Pgsc_voting.score(X_test, y_test)


        metdB=METADES(poolB)
        lcaB=LCA(poolB)
        rankB=Rank(poolB)
        knorauB = KNORAU(poolB)
        kneB = KNORAE(poolB)
        olaB = OLA(poolB)
        singleB = SingleBest(poolB)

        metdB.fit(X_valida, y_valida)
        lcaB.fit(X_valida, y_valida)
        rankB.fit(X_valida, y_valida)
        knorauB.fit(X_valida, y_valida)
        kneB.fit(X_valida, y_valida)
        olaB.fit(X_valida, y_valida)
        singleB.fit(X_valida, y_valida)

        accMetaB.append(metdB.score(X_test, y_test))
        accLCAB.append(lcaB.score(X_test, y_test))
        accRankB.append(rankB.score(X_test, y_test))
        accKUB.append(knorauB.score(X_test, y_test))
        accKEB.append(kneB.score(X_test, y_test))
        accOLAB.append(olaB.score(X_test, y_test))
        accSBB.append(singleB.score(X_test, y_test))
        accVotingBag.append(B)
        print("ok_b")

        metdP = METADES(poolP)
        lcaP = LCA(poolP)
        rankP = Rank(poolP)
        knorauP = KNORAU(poolP)
        kneP = KNORAE(poolP)
        olaP = OLA(poolP)
        singleP = SingleBest(poolP)

        metdP.fit(X_valida,y_valida)
        lcaP.fit(X_valida,y_valida)
        rankP.fit(X_valida,y_valida)
        knorauP.fit(X_valida, y_valida)
        kneP.fit(X_valida, y_valida)
        olaP.fit(X_valida, y_valida)
        singleP.fit(X_valida, y_valida)
            #exit(0)
        accMetaP.append(metdP.score(X_test,y_test))
        accLCAP.append(lcaP.score(X_test,y_test))
        accRankP.append(rankP.score(X_test,y_test))
        accKUP.append(knorauP.score(X_test,y_test))
        accKEP.append(kneP.score(X_test,y_test))
        accOLAP.append(olaP.score(X_test,y_test))
        accSBP.append(singleP.score(X_test,y_test))
        accVotingPgsc.append(P)
        print("ok_p")
        kp,ke=wilcoxon(accKEB,accKEP)
        kp2,ku=wilcoxon(accKUB,accKUP)
        op,ol=wilcoxon(accOLAB,accOLAP)
        sp,sb=wilcoxon(accSBB,accSBP)
        lca,lc=wilcoxon(accLCAB,accLCAP)
        rk,rkb=wilcoxon(accRankB,accRankP)
        met,mt=wilcoxon(accMetaB,accMetaP)
        vot,votc=wilcoxon(accVotingBag,accVotingPgsc)

        print(repeticao)
        print(accKUP)


#    print(i)

    #print(accSBB)
    arq1.write('{};{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{}\n'.format(nome_base,kp,ke,kp2,ku,op,ol,sp,sb, lca,lc, rk,rkb,met,mt,vot,votc))
    arq.write('{};{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{}\n'.format(nome_base,
    100*average(accKUB),100*std(accKUB),100*average(accKUP),100*std(accKUP),
    100*average(accKEB),100*std(accKEB),100*average(accKEP),100*std(accKEP),
    100*average(accOLAB),100*std(accOLAB),100*average(accOLAP),100*std(accOLAP),
    100*average(accSBB),100*std(accSBB),100*average(accSBP),100*std(accSBP),
    100*average(accLCAB),100*std(accLCAB),100*average(accLCAP), 100*std(accLCAP),
    100*average(accRankB), 100*std(accRankB),100*average(accRankP),100*std(accRankP),
    100*average(accMetaB),100*std(accMetaB),100*average(accMetaP),100*std(accMetaP),
    100*average(accVotingBag),100*std(accVotingBag),100*average(accVotingPgsc),100*std(accVotingPgsc)))

    arq2.write('{};{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({})\n'.format(nome_base,
    round(100*average(accKUB),1),round(100*std(accKUB),1),round(100*average(accKUP),1), round(100*std(accKUP),1),
    round(100*average(accKEB),1),round(100*std(accKEB),1),round(100*average(accKEP),1),round(100*std(accKEP),1),
    round(100*average(accOLAB),1),round(100*std(accOLAB),1),round(100*average(accOLAP),1),round(100*std(accOLAP),1),
    round(100*average(accSBB),1),round(100*std(accSBB),1),round(100*average(accSBP),1),round(100*std(accSBP),1),
    round(100*average(accLCAB),1),round(100*std(accLCAB),1), round(100*average(accLCAP),1),round(100*std(accLCAP),1),
    round(100*average(accRankB),1),round(100*std(accRankB),1),round(100*average(accRankP),1),round(100*std(accRankP),1),
    round(100*average(accMetaB),1),round(100*std(accMetaB),1),round(100*average(accMetaP),1),round(100*std(accMetaP),1),
    round(100*average(accVotingBag),1),round(100*std(accVotingBag),1),round(100*average(accVotingPgsc),1),round(100*std(accVotingPgsc),1)))
    arq1.close()
    arq2.close()
    arq.close()

repeticao=1
nome_base='Heart'
#nome_base=sys.argv[1]
caminho_data="/home/projeto/Marcos/GA9/"
caminho_base="/home/projeto/Marcos/Bases2/"
caminho_bags="/home/projeto/Marcos/GA9/"
#caminho_data = "/home/monteiro/Marcos/GA3/"
#caminho_base = "/home/monteiro/Marcos/Bases2/"
#caminho_bags = "/home/monteiro/Marcos/GA3/"
print("nome_base")
classes=[]
selecao("9",3)
#roda(1)




