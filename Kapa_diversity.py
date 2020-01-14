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
#print(nome_base)
#bags_ga=sys.argv[2]
#print(bags_ga.split(','))
#exit(0)
#local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
#local = "/home/projeto/Marcos/Bases3/"
#caminho_base = "/home/projeto/Marcos/Bases2/"
#cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"

nome_base='Wine'
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"

bags_ga="20distdiverlinear_teste_parada_dist"
nome_arq="kappa"

print(nome_base)

def cria_arquivos(nome_arq):
    if os.path.isfile('Selecao' + nome_arq + ".csv") == False or os.path.isfile('Latex' + nome_arq + '.txt') == False:
        arq = open('Selecao' + nome_arq + ".csv", 'a')
        arq1 = open('Wilcoxon' + nome_arq + ".csv", 'a')
        arq2 = open('Selecao_resumo' + nome_arq + ".csv", 'a')
        arq3 = open('Latex' + nome_arq + '.txt', 'a')
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
        x = x + "\\textbf{Problemas} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & " \
                "\\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & " \
                "\\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} \\" + "\\ \n" + "\hline\n"
        arq3.write(x)


    else:
        arq = open('Selecao' + nome_arq + ".csv", 'a')
        arq1 = open('Wilcoxon' + nome_arq + ".csv", 'a')
        arq2 = open('Selecao_resumo' + nome_arq + ".csv", 'a')
        arq3 = open('Latex' + nome_arq + '.txt', 'a')
    return arq, arq1, arq2, arq3

def monta_latex(wil, accVotingBag, accVotingPgsc, accLCAB, accLCAP, accOLAB, accOLAP, accRankB, accRankP, accKEB,
                accKEP, accKUB, accKUP, accMetaB, accMetaP, accSBB, accSBP):
    resultados = [
        round(100 * average(accVotingBag), 1), round(100 * std(accVotingBag), 1),
        round(100 * average(accVotingPgsc), 1), round(100 * std(accVotingPgsc), 1),
        round(100 * average(accLCAB), 1), round(100 * std(accLCAB), 1), round(100 * average(accLCAP), 1),
        round(100 * std(accLCAP), 1),
        round(100 * average(accOLAB), 1), round(100 * std(accOLAB), 1), round(100 * average(accOLAP), 1),
        round(100 * std(accOLAP), 1),
        round(100 * average(accRankB), 1), round(100 * std(accRankB), 1), round(100 * average(accRankP), 1),
        round(100 * std(accRankP), 1),
        round(100 * average(accKEB), 1), round(100 * std(accKEB), 1), round(100 * average(accKEP), 1),
        round(100 * std(accKEP), 1),
        round(100 * average(accKUB), 1), round(100 * std(accKUB), 1), round(100 * average(accKUP), 1),
        round(100 * std(accKUP), 1),
        round(100 * average(accMetaB), 1), round(100 * std(accMetaB), 1), round(100 * average(accMetaP), 1),
        round(100 * std(accMetaP), 1)]
        #round(100 * average(accSBB), 1), round(100 * std(accSBB), 1), round(100 * average(accSBP), 1),
       # round(100 * std(accSBP), 1)]
    br = "\\"
    x = br+"textbf{"+nome_base + "} & "

    cont = 0
    for i in range(0, len(resultados), 4):
       # print(i)
        if wil[cont] < 0.05:
           # print(i)
            if resultados[i] > resultados[i + 2]:
                x = x + br + "textbf{" + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")*} & " + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ") & "
            else:
                x = x  + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")" + " & " +br+ "textbf{" + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ")*} & "
        else:
            if resultados[i] > resultados[i + 2]:
                x = x  + br +"textbf{" + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")} & " + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ") & "
            else:
                x = x + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")" + " & " +br+ "textbf{" + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ")} & "
        if i == len(resultados):
            x = x + br + br
        cont = cont + 1
    x = x + br+br+ "\n"
    return x


def monta_resultados(arq, arq2, accVotingBag, accVotingPgsc, accLCAB, accLCAP, accOLAB, accOLAP, accRankB, accRankP,
                     accKEB, accKEP, accKUB, accKUP, accMetaB, accMetaP, accSBB, accSBP):
    print(accVotingBag)
    print(accMetaP)
    arq.write('{};{};{};{};{};;{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'
              .format(nome_base,

                      100 * average(accVotingBag), 100 * average(accVotingPgsc),
                      100 * average(accLCAB), 100 * average(accLCAP),
                      100 * average(accOLAB),  100 * average(accOLAP),
                      100 * average(accRankB), 100 * average(accRankP),

                      100 * average(accKEB),  100 * average(accKEP),
                      100 * average(accKUB),  100 * average(accKUP),
                      100 * average(accMetaB), 100 * average(accMetaP),

                      100 * average(accSBB), 100 * average(accSBP),

                      100 * std(accVotingPgsc), 100 * std(accVotingBag),
                      100 * std(accLCAB), 100 * std(accLCAP),
                      100 * std(accOLAB), 100 * std(accOLAP),
                      100 * std(accRankB), 100 * std(accRankP),

                      100 * std(accKEB), 100 * std(accKEP),
                      100 * std(accKUB), 100 * std(accKUP),
                      100 * std(accMetaB), 100 * std(accMetaP),

                      100 * std(accSBB), 100 * std(accSBP)))
    arq2.write(
        '{};{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({})\n'.format(
            nome_base,

            round(100 * average(accVotingBag), 1), round(100 * std(accVotingBag), 1),
            round(100 * average(accVotingPgsc), 1), round(100 * std(accVotingPgsc), 1),
            round(100 * average(accLCAB), 1), round(100 * std(accLCAB), 1), round(100 * average(accLCAP), 1),
            round(100 * std(accLCAP), 1),
            round(100 * average(accOLAB), 1), round(100 * std(accOLAB), 1), round(100 * average(accOLAP), 1),
            round(100 * std(accOLAP), 1),
            round(100 * average(accRankB), 1), round(100 * std(accRankB), 1), round(100 * average(accRankP), 1),
            round(100 * std(accRankP), 1),

            round(100 * average(accKEB), 1), round(100 * std(accKEB), 1), round(100 * average(accKEP), 1),
            round(100 * std(accKEP), 1),
            round(100 * average(accKUB), 1), round(100 * std(accKUB), 1), round(100 * average(accKUP), 1),
            round(100 * std(accKUP), 1),
            round(100 * average(accMetaB), 1), round(100 * std(accMetaB), 1), round(100 * average(accMetaP), 1),
            round(100 * std(accMetaP), 1),
            round(100 * average(accSBB), 1), round(100 * std(accSBB), 1), round(100 * average(accSBP), 1),
            round(100 * std(accSBP), 1)))


def selecao(nome_base,local, cpx_caminho,bags_ga,nome_arq,repeticao):
    arq_dataset = local + "Dataset/" + nome_base + ".arff"
    arq_arff = Marff.abre_arff(arq_dataset)
    X, y,_= Marff.retorna_instacias(arq_arff)


    for j in range(1,repeticao):
        print(j)
        poolBag = []
        poolPgsc = []
        predictB = []
        predictP= []
        bags = Cpx.open_bag(cpx_caminho+str(j)+"/", nome_base)
        bags2 = Cpx.open_bag(cpx_caminho+str(j)+"/", nome_base + bags_ga)
        teste, validacao=Cpx.open_test_vali(local,nome_base,j)

        X_test,y_test=Cpx.biuld_x_y(teste,X,y)
        X_valida,y_valida=Cpx.biuld_x_y(validacao,X,y)
        X_test=np.array(X_test)
        X_valida=np.array(X_valida)


        scaler = StandardScaler()
        X_valida = scaler.fit_transform(X_valida)
        X_test = scaler.transform(X_test)

        for i in range(100):

            X_bag,y_bag=Cpx.biuld_x_y(bags['inst'][i],X,y)
            X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
            X_bag = scaler.transform(X_bag)
            X_bag2 = scaler.transform(X_bag2)

            percB = perc.Perceptron(n_jobs=4,tol=1.0)
            percP = perc.Perceptron(n_jobs=4,tol=1.0)

            poolBag.append(percB.fit(X_bag, y_bag))
            poolPgsc.append(percP.fit(X_bag2, y_bags2))



        for i in poolBag:
            predictB.append(i.predict(X_valida))
        for i in poolPgsc:
            predictP.append(i.predict(X_valida))
        kappaB, kappa_stdB,kappa_allB=(Cpx.diversity_kapa(y_valida,predictB))
        kappaP,kappa_stdP,kappa_allP=(Cpx.diversity_kapa(y_valida, predictP))
        save_kappa(nome_base,str(j), "parada_dist_kappa" ,kappa_allB, kappaB,kappa_stdB)
        print(kappa_allB)
        print(len(kappa_allB))
        exit(0)

import warnings

#warnings.filterwarnings("ignore", category=Warning)
def save_kappa(nome_base, iteracao, nome_arq, kappa_all, kappa_mean, kappa_std ):
    with open(nome_base +"_"+iteracao+"_"+ nome_arq+'.csv', 'a', newline='') as csvfile:
          spamwriter = csv.writer(csvfile, delimiter=';')
          for i in range(len(kappa_all)):
                kappa_all[i].append(kappa_mean[i])
                kappa_all[i].append(kappa_std[i])
                spamwriter.writerow(kappa_all[i])


selecao(nome_base,local, cpx_caminho,bags_ga,nome_arq,repeticao=21)

