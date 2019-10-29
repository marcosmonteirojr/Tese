import Marff, Cpx, csv, os
import numpy as np
import novo_perceptron as Nperc
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
import sys, pds_dsoc2
from mlxtend.classifier import EnsembleVoteClassifier
import pds_pool4
#from imblearn.metrics import geometric_mean_score

#nome_base=sys.argv[1]
#bags_ga=sys.argv[2]
#print(bags_ga.split(','))
#exit(0)
#local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
#local = "/home/projeto/Marcos/Bases3/"
#caminho_base = "/home/projeto/Marcos/Bases2/"
#cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"

nome_base='Banana'
local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
bags_ga="20ep"

nome_arq="teste"


arq_dataset = caminho_base + "Dataset/" + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
X, y, data = Marff.retorna_instacias(arq_arff)



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

accDsocp=[]
accDsocb=[]

accVotingBag = []
accVotingPgsc = []

for j in range(1,3):

    poolBag = []
    poolPgsc = []
    calibrated_poolBag = []
    calibrated_poolP = []

    bags = Cpx.open_bag(cpx_caminho+str(j)+"/", nome_base)
    bags2 = Cpx.open_bag(cpx_caminho+str(j)+"/", nome_base + bags_ga)
    teste, validacao=Cpx.open_test_vali(local,nome_base,j)

    X_test,y_test=Cpx.biuld_x_y(teste,X,y)
    X_valida,y_valida=Cpx.biuld_x_y(validacao,X,y)
    X_test=np.array(X_test)
    X_valida=np.array(X_valida)

    for i in range(100):

        X_bag,y_bag=Cpx.biuld_x_y(bags['inst'][i],X,y)
        X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)

       # print(X_bag2[1],"\n")
        #X_bag = scaler.transform(X_bag)
        #print(X_bag[0])
        #X_bag2 = scaler.transform(X_bag2)
        #
        percB = perc.Perceptron(n_jobs=4,max_iter=100,tol=10.0)
        percP = perc.Perceptron(n_jobs=4,max_iter=100,tol=10.0)

        poolBag.append(percB.fit(X_bag, y_bag))
        poolPgsc.append(percP.fit(X_bag2, y_bags2))


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

kp,ke=wilcoxon(accKEB,accKEP)
kp2,ku=wilcoxon(accKUB,accKUP)
op,ol=wilcoxon(accOLAB,accOLAP)
sp,sb=wilcoxon(accSBB,accSBP)
lca,lc=wilcoxon(accLCAB,accLCAP)
rk,rkb=wilcoxon(accRankB,accRankP)
met,mt=wilcoxon(accMetaB,accMetaP)
vot,votc=wilcoxon(accVotingBag,accVotingPgsc)
dsc,dsoc=wilcoxon(accDsocb,accDsocp)
wil=[ku,ke,ol,sb,lc,rkb,mt,vot]



if os.path.isfile('SelecaoMedia_desvio' + nome_arq + ".csv") == False or os.path.isfile('SelecaoLatex' + nome_arq + '.txt') == False:
    print("kjksjdf")
    arq = open('SelecaoMedia_desvio' + nome_arq + ".csv", 'a')
    arq1 = open('SelecaoWilcoxon' + nome_arq + ".csv", 'a')
    arq2 = open('SelecaoTabela' + nome_arq + ".csv", 'a')
    arq3 = open('SelecaoLatex' + nome_arq + '.txt', 'a')
    arq.write(';ALL;;LCA;;OLA;;Rank;;Knora-E;;Knora-U;;Meta;;SB\n')
    arq.write(';Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X\n')
    arq1.write(';ALL;;LCA;;OLA;;Rank;;Knora-E;;Knora-U;;Meta;;SB\n')
    arq1.write(';Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X\n')
    arq2.write(';ALL;;LCA;;OLA;;Rank;;Knora-E;;Knora-U;;Meta;;SB\n')
    arq2.write(';Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X\n')

    x = "\\begin{table}[]\n"
    x = x + "\\begin{tabular}{ccccccccccccccc}\n\hline\n"
    x = x + "\\textbf{} & \multicolumn{ 2}{c}{\\textbf{Voto majoritário}} & \multicolumn{ 2}{c}{\\textbf{LCA}} & \multicolumn{ 2}{c}" \
            "{\\textbf{OLA}} & \multicolumn{ 2}{c}{\\textbf{Rank}} & \multicolumn{ 2}{c}" \
            "{\\textbf{Knora-E}} & \multicolumn{ 2}{c}{\\textbf{Knora-U}} & \multicolumn{ 2}{c}{\\textbf{Meta-Des}} \\" + "\\ \n \hline\n"
    x = x+ "\\textbf{Problemas} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & " \
        "\\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & " \
        "\\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} \\" + "\\ \n" + "\hline\n"
    arq3.write(x)

else:
    arq = open('SelecaoMedia_desvio' + nome_arq + ".csv", 'a')
    arq1 = open('SelecaoWilcoxon' + nome_arq + ".csv", 'a')
    arq2 = open('SelecaoTabela' + nome_arq + ".csv", 'a')
    arq3 = open('SelecaoLatex' + nome_arq + '.txt', 'a')

def monta_string():
    resultados = [
        round(100 * average(accVotingBag), 1), round(100 * std(accVotingBag), 1),round(100 * average(accVotingPgsc), 1), round(100 * std(accVotingPgsc), 1),
        round(100 * average(accLCAB), 1), round(100 * std(accLCAB), 1), round(100 * average(accLCAP), 1),round(100 * std(accLCAP), 1),
        round(100 * average(accOLAB), 1), round(100 * std(accOLAB), 1), round(100 * average(accOLAP), 1),round(100 * std(accOLAP), 1),
        round(100 * average(accRankB), 1), round(100 * std(accRankB), 1), round(100 * average(accRankP), 1),round(100 * std(accRankP), 1),

        round(100 * average(accKEB), 1), round(100 * std(accKEB), 1), round(100 * average(accKEP), 1), round(100 * std(accKEP),1),
        round(100 * average(accKUB), 1), round(100 * std(accKUB), 1), round(100 * average(accKUP), 1),round(100 * std(accKUP), 1),
        round(100 * average(accMetaB), 1), round(100 * std(accMetaB), 1), round(100 * average(accMetaP), 1),round(100 * std(accMetaP), 1)]

        #round(100 * average(accSBB), 1), round(100 * std(accSBB), 1), round(100 * average(accSBP), 1), round(100 * std(accSBP),1)]




    x =  nome_base+"&"
    cont=0

    for i in range(0,len(resultados), 4):
        print(i)
        if wil[cont]<0.05:
            print(i)

            if resultados[i] > resultados[i + 2]:
                x = x + '\\'+"\\"+"textbf{" + str(resultados[i]) + "}(" + str(resultados[i + 1]) + ")*&" + str(resultados[i + 2]) + "(" + str(resultados[i + 3]) + ")"
            else:
                x = x + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")" + "&\\"+"textbf{" + str(resultados[i + 2]) + "}(" + str(resultados[i + 3]) + ")*"
        else:
            if resultados[i] > resultados[i + 2]:
                x = x + "&\\"+"textbf{" + str(resultados[i]) + "}(" + str(resultados[i + 1]) + ")&" + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ")\\"+"\\"
            else:
                x = x + "&"+str(resultados[i]) + "(" + str(resultados[i + 1]) + ")" + "&\\"+"textbf{" + str(
                    resultados[i + 2]) + "}(" + str(resultados[i + 3]) + ")\\"+"\\"
        if i == len(resultados):
            x=x+'\\'+"\\"
        cont=cont+1
    x=x+ "\n"

    return x
x=monta_string()
arq3.write(x)
exit(0)
arq1.write('{};{};{};{};{};;{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(nome_base,vot,votc,lca,lc,op,ol,rk,rkb,kp,ke,kp2,ku,met,mt,sp,sb))
arq.write('{};{};{};{};{};;{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'
.format(nome_base,

100*average(accVotingBag),100*std(accVotingBag),100*average(accVotingPgsc),100*std(accVotingPgsc),
100*average(accLCAB),100*std(accLCAB),100*average(accLCAP), 100*std(accLCAP),
100*average(accOLAB),100*std(accOLAB),100*average(accOLAP),100*std(accOLAP),
100*average(accRankB), 100*std(accRankB),100*average(accRankP),100*std(accRankP),

100*average(accKEB),100*std(accKEB),100*average(accKEP),100*std(accKEP),
100*average(accKUB),100*std(accKUB),100*average(accKUP),100*std(accKUP),
100*average(accMetaB),100*std(accMetaB),100*average(accMetaP),100*std(accMetaP),

100*average(accSBB),100*std(accSBB),100*average(accSBP),100*std(accSBP)))

arq2.write('{};{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({}))\n'.format(nome_base,

round(100*average(accVotingBag),1),round(100*std(accVotingBag),1),round(100*average(accVotingPgsc),1),round(100*std(accVotingPgsc),1),
round(100*average(accLCAB),1),round(100*std(accLCAB),1), round(100*average(accLCAP),1),round(100*std(accLCAP),1),
round(100*average(accOLAB),1),round(100*std(accOLAB),1),round(100*average(accOLAP),1),round(100*std(accOLAP),1),
round(100*average(accRankB),1),round(100*std(accRankB),1),round(100*average(accRankP),1),round(100*std(accRankP),1),


round(100*average(accKEB),1),round(100*std(accKEB),1),round(100*average(accKEP),1),round(100*std(accKEP),1),
round(100*average(accKUB),1),round(100*std(accKUB),1),round(100*average(accKUP),1), round(100*std(accKUP),1),
round(100*average(accMetaB),1),round(100*std(accMetaB),1),round(100*average(accMetaP),1),round(100*std(accMetaP),1),

round(100*average(accSBB),1),round(100*std(accSBB),1),round(100*average(accSBP),1),round(100*std(accSBP),1)))


arq1.close()
arq2.close()
arq3.close()
arq.close()

