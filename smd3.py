import Marff, Cpx, csv
import numpy as np
#import novo_perceptron as perc
from deslib.static.single_best import SingleBest
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import perceptron as perc
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.dcs.ola import OLA
from deslib.des.meta_des import METADES
from deslib.dcs.lca import LCA
from deslib.dcs.rank import Rank
from scipy.stats import wilcoxon
from numpy import average,std, array, argsort
import sys
from mlxtend.classifier import EnsembleVoteClassifier

#nome_base=sys.argv[1]
#
#print(bags_ga)
#exit(0)

nome_base='Wine'
local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bagsx/"
bags_ga="20sgc"
#min_score=0

#local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
#local = "/home/projeto/Marcos/Bases3/"
#caminho_base = "/home/projeto/Marcos/Bases2/"
#cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"


arq_dataset = caminho_base + "Dataset/" + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
X, y, _ = Marff.retorna_instacias(arq_arff)




arq = open('SelecaoMedia_desvio_pgcs_gax.csv', 'a')
arq1 = open('SelecaoWilcoxon_pgcs_gax.csv', 'a')
arq2 = open('SelecaoTabela_pgcs_gax.csv', 'a')
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

for j in range(1,21):

    poolBag = []
    poolPgsc = []
    calibrated_poolBag = []
    calibrated_poolP = []

    bags = Cpx.open_bag(cpx_caminho+str(j)+"/", nome_base)
    #print(cpx_caminho+str(j)+"/", nome_base + "20sc")
    bags2 = Cpx.open_bag(cpx_caminho+str(j)+"/", nome_base + bags_ga)
    #print(bags)
    #exit(0)
    teste, validacao=Cpx.open_test_vali(local,nome_base,j)

    X_test,y_test=Cpx.biuld_x_y(teste,X,y)
    X_valida,y_valida=Cpx.biuld_x_y(validacao,X,y)
    X_test=np.array(X_test)
    X_valida=np.array(X_valida)

    y_test = np.array(y_test)
    y_valida = np.array(y_valida)

    scaler = StandardScaler()
    sca=StandardScaler()
    X_valida = scaler.fit_transform(X_valida)
    X_test = sca.fit_transform(X_test)

    for i in range(100):
       # print(i)
        X_bag,y_bag=Cpx.biuld_x_y(bags['inst'][i],X,y)
       # print(X_bag[1])
        X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
        #print(X_bag2[1],"\n")
        X_bag = scaler.transform(X_bag)
        #print(X_bag[0])
        X_bag2 = scaler.transform(X_bag2)
        #
        percB = perc.Perceptron(n_jobs=4,max_iter=100,tol=10.0)
        percP = perc.Perceptron(n_jobs=4,max_iter=100,tol=10.0)

        poolBag.append(percB.fit(X_bag, y_bag))
        poolPgsc.append(percP.fit(X_bag2, y_bags2))

    orc=Cpx.oracle(poolBag,X_valida,y_valida,X_test,y_test)

    #print(orc)
    #exit(0)
    for clf in poolBag:
        calibrated = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
        calibrated.fit(X_valida, y_valida)
        calibrated_poolBag.append(calibrated)
    for clf in poolPgsc:
        calibrated = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
        calibrated.fit(X_valida, y_valida)
        calibrated_poolP.append(calibrated)


    bagging_voting = EnsembleVoteClassifier(clfs=poolBag, voting='hard', refit=False)
    bagging_vot = bagging_voting.fit(X_valida, y_valida)
    B =  bagging_vot.score(X_test, y_test)

    Pgsc_voting = EnsembleVoteClassifier(clfs=poolPgsc, voting='hard', refit=False)
    Pgsc_voting = Pgsc_voting.fit(X_valida, y_valida)
    P =  Pgsc_voting.score(X_test, y_test)

    metdb = METADES(calibrated_poolBag)
    metdp=METADES(calibrated_poolP)
    #exit(0)
    lcab=LCA(poolBag)
    lcap=LCA(poolPgsc)
    rankb=Rank(poolBag)
    rankp=Rank(poolPgsc)
    knorauB = KNORAU(poolBag)
    knorauP = KNORAU(poolPgsc)
    kneB = KNORAE(poolBag)
    kneP = KNORAE(poolPgsc)
    olaB = OLA(poolBag)
    olaP = OLA(poolPgsc)
    singleP = SingleBest(poolPgsc)
    singleB = SingleBest(poolBag)

    #
    metdb.fit(X_valida, y_valida)
    metdp.fit(X_valida, y_valida)
    lcab.fit(X_valida, y_valida)
    lcap.fit(X_valida, y_valida)
    rankb.fit(X_valida, y_valida)
    rankp.fit(X_valida, y_valida)

    knorauB.fit(X_valida, y_valida)
    knorauP.fit(X_valida, y_valida)
    kneB.fit(X_valida, y_valida)
    kneP.fit(X_valida, y_valida)
    olaB.fit(X_valida, y_valida)
    olaP.fit(X_valida, y_valida)

    singleP.fit(X_valida, y_valida)
    singleB.fit(X_valida, y_valida)

    #     #exit(0)
    accMetaB.append(metdb.score(X_test,y_test))
    accMetaP.append(metdp.score(X_test,y_test))
    accLCAB.append(lcab.score(X_test,y_test))
    accLCAP.append(lcap.score(X_test,y_test))
    accRankB.append(rankb.score(X_test,y_test))
    accRankP.append(rankp.score(X_test,y_test))
    accKUB.append(knorauB.score(X_test,y_test))
    accKUP.append(knorauP.score(X_test, y_test))
    accKEB.append(kneB.score(X_test,y_test))
    accKEP.append(kneP.score(X_test, y_test))
    accOLAB.append(olaB.score(X_test,y_test))
    accOLAP.append(olaP.score(X_test, y_test))
    accSBP.append(singleP.score(X_test, y_test))
    accSBB.append(singleB.score(X_test, y_test))

    accVotingBag.append(B)
    accVotingPgsc.append(P)
    print(j)
#exit(0)

    #     singleM = SingleBest(poolPgsc)
    #     #

    #     #

    #     #
kp,ke=wilcoxon(accKEB,accKEP)
kp2,ku=wilcoxon(accKUB,accKUP)
op,ol=wilcoxon(accOLAB,accOLAP)
sp,sb=wilcoxon(accSBB,accSBP)
lca,lc=wilcoxon(accLCAB,accLCAP)
rk,rkb=wilcoxon(accRankB,accRankP)
met,mt=wilcoxon(accMetaB,accMetaP)
vot,votc=wilcoxon(accVotingBag,accVotingPgsc)
    #

arq1.write('{};{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{}\n'.format(nome_base,kp,ke,kp2,ku,op,ol,sp,sb, lca,lc, rk,rkb,met,mt,vot,votc))
arq.write('{};{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{}\n'
.format(nome_base,100*average(accKUB),100*std(accKUB),100*average(accKUP),100*std(accKUP),
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


