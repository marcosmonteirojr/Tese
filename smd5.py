import Marff, Cpx, csv, os, sys
import numpy as np
import novo_perceptron
from deslib.static.single_best import SingleBest
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron as perc
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.dcs.ola import OLA
from deslib.des.meta_des import METADES
from deslib.dcs.lca import LCA
from deslib.dcs.rank import Rank
from scipy.stats import wilcoxon
from numpy import average, std
from mlxtend.classifier import EnsembleVoteClassifier

# import pds_pool4
# from imblearn.metrics import geometric_mean_score
# nome_base="Lithuanian"
#nome_base = "Wine"
# print(nome_base)
nome_base=sys.argv[1]
# print(bags_ga.split(','))
# exit(0)
local = "/home/marcosmonteiro/Marcos/Bases3/"
# local = "/media/marcos/Data/Tese/Bases3/"
# caminho_base = "/home/marcosmonteiro/Marcos/Bases32/"
cpx_caminho="/home/marcosmonteiro/Marcos/Bases3/Bags/tree/Bags/"
# nome_base="P2"
#local = "/media/marcos/Data/Tese/Bases3/"
#cpx_caminho = "/media/marcos/Data/Tese/Bases3/Bags/tree/Bags/"
# local ="/home/projeto/Marcos/Bases3/"
bags_ga = "19maxacctree"
bags_ga2= "19maxdistancetree"
nome_arq = "resutados_tree"
accKUB = []
accKEB = []
accOLAB = []
accSaccKUB = []
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

accVotingPool = []
accVotingPgsc = []

accKUP2 = []
accKEP2 = []
accOLAP2 = []
accSBP2 = []

accLCAP2 = []
accRankP2 = []
accMetaP2 = []

accVotingPgsc2 = []


def cria_arquivo_individual(nome_arq, tipo, accVotingPgsc, accVotingPool, accLCAP,
                            accLCAB, accOLAP, accOLAB, accRankP, accRankB, accKEP, accKEB, accKUP, accKUB, accMetaP,
                            accMetaB, accVotingPgsc2, accLCAP2, accOLAP2, accRankP2, accKEP2, accKUP2, accMetaP2):
    MVR_ = open('Selecao_MVR' + nome_arq + ".csv", 'a')
    LCA_ = open('Selecao_LCA' + nome_arq + ".csv", 'a')
    OLA_ = open('Selecao_OLA' + nome_arq + ".csv", 'a')
    RANK_ = open('Selecao_RANK' + nome_arq + ".csv", 'a')
    KNORA_E = open('Selecao_KE' + nome_arq + ".csv", 'a')
    KNORA_U = open('Selecao_KU' + nome_arq + ".csv", 'a')
    META_ = open('Selecao_META' + nome_arq + ".csv", 'a')
    MVR_2 = open('Selecao_MVR_n' + nome_arq + ".csv", 'a')
    LCA_2 = open('Selecao_LCA_n' + nome_arq + ".csv", 'a')
    OLA_2 = open('Selecao_OLA_n' + nome_arq + ".csv", 'a')
    RANK_2 = open('Selecao_RANK_n' + nome_arq + ".csv", 'a')
    KNORA_E2 = open('Selecao_KE_n' + nome_arq + ".csv", 'a')
    KNORA_U2 = open('Selecao_KU_n' + nome_arq + ".csv", 'a')
    META_2 = open('Selecao_META_n' + nome_arq + ".csv", 'a')

    if accVotingPgsc2 and tipo == 'bagging':
        MVR_.write('{};{}({});{}({});{}({});'.format(nome_base, round(100 * average(accVotingPgsc), 1),
                                                     round(100 * std(accVotingPgsc), 1),
                                                     round(100 * average(accVotingPgsc2), 1),
                                                     round(100 * std(accVotingPgsc2), 1),
                                                     round(100 * average(accVotingPool), 1),
                                                     round(100 * std(accVotingPool), 1)))
        MVR_2.write('{};{};{};{};'.format(nome_base, round(100 * average(accVotingPgsc), 1),
                                          round(100 * average(accVotingPgsc2), 1),
                                          round(100 * average(accVotingPool), 1)))
        LCA_.write('{};{}({});{}({});{}({});'.format(nome_base, round(100 * average(accLCAP), 1),
                                                     round(100 * std(accLCAP), 1),
                                                     round(100 * average(accLCAP2), 1),
                                                     round(100 * std(accLCAP2), 1),
                                                     round(100 * average(accLCAB), 1),
                                                     round(100 * std(accLCAB), 1)))
        LCA_2.write('{};{};{};{};'.format(nome_base, round(100 * average(accLCAP), 1),
                                          round(100 * average(accLCAP2), 1),
                                          round(100 * average(accLCAB), 1)))
        OLA_.write('{};{}({});{}({});{}({});'.format(nome_base, round(100 * average(accOLAP), 1),
                                                     round(100 * std(accOLAP), 1),
                                                     round(100 * average(accOLAP2), 1),
                                                     round(100 * std(accOLAP2), 1),
                                                     round(100 * average(accOLAB), 1),
                                                     round(100 * std(accOLAB), 1)))
        OLA_2.write('{};{};{};{};'.format(nome_base, round(100 * average(accOLAP), 1),
                                          round(100 * average(accOLAP2), 1),
                                          round(100 * average(accOLAB), 1)))
        RANK_.write('{};{}({});{}({});{}({});'.format(nome_base, round(100 * average(accRankP), 1),
                                                      round(100 * std(accRankP), 1),
                                                      round(100 * average(accRankP2), 1),
                                                      round(100 * std(accRankP2), 1),
                                                      round(100 * average(accRankB), 1),
                                                      round(100 * std(accRankB), 1)))
        RANK_2.write('{};{};{};{};'.format(nome_base, round(100 * average(accRankP), 1),
                                           round(100 * average(accRankP2), 1),
                                           round(100 * average(accRankB), 1)))
        KNORA_E.write('{};{}({});{}({});{}({});'.format(nome_base, round(100 * average(accKEP), 1),
                                                        round(100 * std(accKEP), 1),
                                                        round(100 * average(accKEP2), 1),
                                                        round(100 * std(accKEP2), 1),
                                                        round(100 * average(accKEB), 1),
                                                        round(100 * std(accKEB), 1)))
        KNORA_E2.write('{};{};{};{};'.format(nome_base, round(100 * average(accKEP), 1),
                                             round(100 * average(accKEP2), 1),
                                             round(100 * average(accKEB), 1)))
        KNORA_U.write('{};{}({});{}({});{}({});'.format(nome_base, round(100 * average(accKUP), 1),
                                                        round(100 * std(accKUP), 1),
                                                        round(100 * average(accKUP2), 1),
                                                        round(100 * std(accKUP2), 1),
                                                        round(100 * average(accKUB), 1),
                                                        round(100 * std(accKUB), 1)))
        KNORA_U2.write('{};{};{};{};'.format(nome_base, round(100 * average(accKUP), 1),
                                             round(100 * average(accKUP2), 1),
                                             round(100 * average(accKUB), 1)))

        META_.write('{};{}({});{}({});{}({});'.format(nome_base, round(100 * average(accMetaP), 1),
                                                      round(100 * std(accMetaP), 1),
                                                      round(100 * average(accMetaP2), 1),
                                                      round(100 * std(accMetaP2), 1),
                                                      round(100 * average(accMetaB), 1),
                                                      round(100 * std(accMetaB), 1)))
        META_2.write('{};{};{};{};'.format(nome_base, round(100 * average(accMetaP), 1),
                                           round(100 * average(accMetaP2), 1),
                                           round(100 * average(accMetaB), 1)))

    elif accVotingPgsc2 == None and tipo == 'bagging':

        MVR_.write('{};{}({});{}({});{}({});;'.format(nome_base, round(100 * average(accVotingPgsc), 1),
                                                      round(100 * std(accVotingPgsc), 1),
                                                      round(100 * average(accVotingPool), 1),
                                                      round(100 * std(accVotingPool), 1)))
        MVR_2.write('{};{};{};'.format(nome_base, round(100 * average(accVotingPgsc), 1),
                                       round(100 * average(accVotingPool), 1)))
        LCA_.write('{};{}({});{};'.format(nome_base, round(100 * average(accLCAP), 1),
                                          round(100 * std(accLCAP), 1),
                                          round(100 * average(accLCAB), 1),
                                          round(100 * std(accLCAB), 1)))
        LCA_2.write('{};{};{};'.format(nome_base, round(100 * average(accLCAP), 1),
                                       round(100 * average(accLCAB), 1)))
        OLA_.write('{};{}({});{}({});'.format(nome_base, round(100 * average(accOLAP), 1),
                                              round(100 * std(accOLAP), 1),
                                              round(100 * average(accOLAB), 1),
                                              round(100 * std(accOLAB), 1)))
        OLA_2.write('{};{};{};'.format(nome_base, round(100 * average(accOLAP), 1),
                                       round(100 * average(accOLAB), 1)))
        RANK_.write('{};{}({});{}({});'.format(nome_base, round(100 * average(accRankP), 1),
                                               round(100 * std(accRankP), 1),
                                               round(100 * average(accRankB), 1),
                                               round(100 * std(accRankB), 1)))
        RANK_2.write('{};{};{};'.format(nome_base, round(100 * average(accRankP), 1),
                                        round(100 * average(accRankB), 1)))
        KNORA_E.write('{};{}({});{}({});'.format(nome_base, round(100 * average(accKEP), 1),
                                                 round(100 * std(accKEP), 1),
                                                 round(100 * average(accKEB), 1),
                                                 round(100 * std(accKEB), 1)))
        KNORA_E2.write('{};{};{};'.format(nome_base, round(100 * average(accKEP), 1),
                                          round(100 * average(accKEB), 1)))
        KNORA_U.write('{};{}({});{}({});'.format(nome_base, round(100 * average(accKUP), 1),
                                                 round(100 * std(accKUP), 1),
                                                 round(100 * average(accKUB), 1),
                                                 round(100 * std(accKUB), 1)))
        KNORA_U2.write('{};{};{};'.format(nome_base, round(100 * average(accKUP), 1),
                                          round(100 * average(accKUB), 1)))

        META_.write('{};{}({});{}({});'.format(nome_base, round(100 * average(accMetaP), 1),
                                               round(100 * std(accMetaP), 1),
                                               round(100 * average(accMetaB), 1),
                                               round(100 * std(accMetaB), 1)))
        META_2.write('{};{};{};'.format(nome_base, round(100 * average(accMetaP), 1),
                                        round(100 * average(accMetaB), 1)))
    elif tipo == 'ada':
        MVR_.write('{}({});'.format(round(100 * average(accVotingPool), 1), round(100 * std(accVotingPool), 1)))
        LCA_.write('{}({});'.format(round(100 * average(accLCAB), 1),
                                    round(100 * std(accLCAB), 1)))
        OLA_.write('{}({});'.format(round(100 * average(accOLAB), 1),
                                    round(100 * std(accOLAB), 1)))
        RANK_.write('{}({});'.format(round(100 * average(accRankB), 1),
                                     round(100 * std(accRankB), 1)))
        KNORA_E.write('{}({});'.format(round(100 * average(accKEB), 1),
                                       round(100 * std(accKEB), 1)))
        KNORA_U.write('{}({});'.format(round(100 * average(accKUB), 1),
                                       round(100 * std(accKUB), 1)))
        META_.write('{}({});'.format(round(100 * average(accMetaB), 1),
                                     round(100 * std(accMetaB), 1)))

        MVR_2.write('{};'.format(round(100 * average(accVotingPool), 1)))
        LCA_2.write('{};'.format(round(100 * average(accLCAB), 1)))
        OLA_2.write('{};'.format(round(100 * average(accOLAB), 1)))
        RANK_2.write('{};'.format(round(100 * average(accRankB), 1)))
        KNORA_E2.write('{};'.format(round(100 * average(accKEB), 1)))
        KNORA_U2.write('{};'.format(round(100 * average(accKUB), 1)))
        META_2.write('{};'.format(round(100 * average(accMetaB), 1)))

    elif tipo == 'rf':
        MVR_.write('{}({})\n'.format(round(100 * average(accVotingPool), 1), round(100 * std(accVotingPool), 1)))
        LCA_.write('{}({})\n'.format(round(100 * average(accLCAB), 1),
                                     round(100 * std(accLCAB), 1)))
        OLA_.write('{}({})\n'.format(round(100 * average(accOLAB), 1),
                                     round(100 * std(accOLAB), 1)))
        RANK_.write('{}({})\n'.format(round(100 * average(accRankB), 1),
                                      round(100 * std(accRankB), 1)))
        KNORA_E.write('{}({})\n'.format(round(100 * average(accKEB), 1),
                                        round(100 * std(accKEB), 1)))
        KNORA_U.write('{}({})\n'.format(round(100 * average(accKUB), 1),
                                        round(100 * std(accKUB), 1)))
        META_.write('{}({})\n'.format(round(100 * average(accMetaB), 1),
                                      round(100 * std(accMetaB), 1)))
        MVR_2.write('{}\n'.format(round(100 * average(accVotingPool), 1)))
        LCA_2.write('{}\n'.format(round(100 * average(accLCAB), 1)))
        OLA_2.write('{}\n'.format(round(100 * average(accOLAB), 1)))
        RANK_2.write('{}\n'.format(round(100 * average(accRankB), 1)))
        KNORA_E2.write('{}\n'.format(round(100 * average(accKEB), 1)))
        KNORA_U2.write('{}\n'.format(round(100 * average(accKUB), 1)))
        META_2.write('{}\n'.format(round(100 * average(accMetaB), 1)))
        MVR_2.close()
        MVR_.close()
        LCA_.close()
        LCA_2.close()
        OLA_.close()
        OLA_2.close()
        RANK_.close()
        RANK_2.close()
        KNORA_U.close()
        KNORA_E.close()
        KNORA_U2.close()
        KNORA_E2.close()
        META_.close()
        META_2.close()


def cria_arquivos(nome_arq):
    if os.path.isfile('Selecao' + nome_arq + ".csv") == False or os.path.isfile('Latex' + nome_arq + '.txt') == False:
        # arq = open('Selecao' + nome_arq + ".csv", 'a')
        # arq1 = open('Wilcoxon' + nome_arq + ".csv", 'a')
        arq2 = open('Selecao_resumo' + nome_arq + ".csv", 'a')
        # arq3 = open('Latex' + nome_arq + '.txt', 'a')
        # arq.write(';ALL;;LCA;;OLA;;Rank;;Knora-E;;Knora-U;;Meta;;SB\n')
        # arq.write(';Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X\n')
        # arq1.write(';ALL;;LCA;;OLA;;Rank;;Knora-E;;Knora-U;;Meta;;SB\n')
        # arq1.write(';Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X\n')
        arq2.write(';ALL;;LCA;;OLA;;Rank;;Knora-E;;Knora-U;;Meta;;SB\n')
        arq2.write(';Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X;Bag;X\n')

    # x = "\\begin{table}[]\n"
    # x = x + "\\begin{tabular}{ccccccccccccccc}\n\hline\n"
    # x = x + "\\textbf{} & \multicolumn{ 2}{c}{\\textbf{Voto majoritário}} & \multicolumn{ 2}{c}{\\textbf{LCA}} & \multicolumn{ 2}{c}" \
    #       "{\\textbf{OLA}} & \multicolumn{ 2}{c}{\\textbf{Rank}} & \multicolumn{ 2}{c}" \
    #       "{\\textbf{Knora-E}} & \multicolumn{ 2}{c}{\\textbf{Knora-U}} & \multicolumn{ 2}{c}{\\textbf{Meta-Des}} \\" + "\\ \n \hline\n"
    # x = x + "\\textbf{Problemas} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & " \
    # "\\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} & \\textbf{Bagging} & " \
    # "\\textbf{Método} & \\textbf{Bagging} & \\textbf{Método} \\" + "\\ \n" + "\hline\n"
    # arq3.write(x)
    else:
        # arq = open('Selecao' + nome_arq + ".csv", 'a')
        #  arq1 = open('Wilcoxon' + nome_arq + ".csv", 'a')
        arq2 = open('Selecao_resumo' + nome_arq + ".csv", 'a')
        # arq3 = open('Latex' + nome_arq + '.txt', 'a')
    return arq2


def monta_latex(wil, accVotingBag, accVotingPgsc, accLCAB, accLCAP, accOLAB, accOLAP, accRankB, accRankP, accKEB,
                accKEP, accKUB, accKUP, accMetaB, accMetaP):
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

    br = "\\"
    x = br + "textbf{" + nome_base + "} & "

    cont = 0
    for i in range(0, len(resultados), 4):
        # print(i)
        if wil[cont] < 0.05:
            # print(i)
            if resultados[i] > resultados[i + 2]:
                x = x + br + "textbf{" + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")*} & " + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ") & "
            else:
                x = x + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")" + " & " + br + "textbf{" + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ")*} & "
        else:
            if resultados[i] > resultados[i + 2]:
                x = x + br + "textbf{" + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")} & " + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ") & "
            else:
                x = x + str(resultados[i]) + "(" + str(resultados[i + 1]) + ")" + " & " + br + "textbf{" + str(
                    resultados[i + 2]) + "(" + str(resultados[i + 3]) + ")} & "
        if i == len(resultados):
            x = x + br + br
        cont = cont + 1
    x = x + br + br + "\n"
    return x


def monta_resultados(arq2, accVotingPgsc, accVotingPool, accLCAP, accLCAB, accOLAP, accOLAB, accRankP, accRankB, accKEP,
                     accKEB, accKUP, accKUB, accMetaP, accMetaB, accVotingPgsc2, accLCAP2, accOLAP2, accRankP2, accKEP2,
                     accKUP2, accMetaP2):
    if accVotingPgsc2:
        arq2.write(
            '{};{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({})\n'.format(
                nome_base, round(100 * average(accVotingPgsc), 1),
                round(100 * std(accVotingPgsc), 1),
                round(100 * average(accVotingPgsc2), 1),
                round(100 * std(accVotingPgsc2), 1),
                round(100 * average(accVotingPool), 1),
                round(100 * std(accVotingPool), 1),
                round(100 * average(accLCAP), 1),
                round(100 * std(accLCAP), 1),
                round(100 * average(accLCAP2), 1),
                round(100 * std(accLCAP2), 1),
                round(100 * average(accLCAB), 1),
                round(100 * std(accLCAB), 1),
                round(100 * average(accOLAP), 1),
                round(100 * std(accOLAP), 1),
                round(100 * average(accOLAP2), 1),
                round(100 * std(accOLAP2), 1),
                round(100 * average(accOLAB), 1),
                round(100 * std(accOLAB), 1),
                round(100 * average(accRankP), 1),
                round(100 * std(accRankP), 1),
                round(100 * average(accRankP2), 1),
                round(100 * std(accRankP2), 1),
                round(100 * average(accRankB), 1),
                round(100 * std(accRankB), 1),
                round(100 * average(accKEP), 1),
                round(100 * std(accKEP), 1),
                round(100 * average(accKEP2), 1),
                round(100 * std(accKEP2), 1),
                round(100 * average(accKEB), 1),
                round(100 * std(accKEB), 1),
                round(100 * average(accKUP), 1),
                round(100 * std(accKUP), 1),
                round(100 * average(accKUP2), 1),
                round(100 * std(accKUP2), 1),
                round(100 * average(accKUB), 1),
                round(100 * std(accKUB), 1),
                round(100 * average(accMetaP), 1),
                round(100 * std(accMetaP), 1),
                round(100 * average(accMetaP2), 1),
                round(100 * std(accMetaP2), 1),
                round(100 * average(accMetaB), 1),
                round(100 * std(accMetaB), 1)))
    else:
        arq2.write(
            '{};{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({});{} ({})\n'.format(
                nome_base,

                round(100 * average(accVotingPool), 1), round(100 * std(accVotingPool), 1),
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
                round(100 * std(accMetaP), 1)))


def arquivos(nome_arq, accVotingPgsc, accVotingPool, accLCAP, accLCAB, accOLAP, accOLAB, accRankP, accRankB, accKEP,
             accKEB, accKUP, accKUB, accMetaP, accMetaB, accVotingPgsc2, accLCAP2, accOLAP2, accRankP2, accKEP2,
             accKUP2, accMetaP2):
    #    lista_final_met = list(set(accMetaP) - set(accMetaB))
    #    lista_final_vot = list(set(accVotingPgsc) - set(accVotingPool))
    #    kp, ke = wilcoxon(accKEB, accKEP)
    #    kp2, ku = wilcoxon(accKUB, accKUP)
    #    op, ol = wilcoxon(accOLAB, accOLAP)
    #    lca, lc = wilcoxon(accLCAB, accLCAP)
    #    rk, rkb = wilcoxon(accRankB, accRankP)
    #    if lista_final_met != []:
    #        met, mt = wilcoxon(accMetaB, accMetaP)
    #    else:
    #        mt = met = 1000
    #    if lista_final_vot != []:
    #        vot, votc = wilcoxon(accVotingPool, accVotingPgsc)
    #   else:
    #       vot = votc = 1000
    #   wil = [ku, ke, ol, lc, rkb, mt, votc]
    arq2 = cria_arquivos(nome_arq)

    #   arq1.write(
    #       '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(nome_base, vot, votc, lca, lc, op, ol, rk, rkb,
    #                                                                    kp, ke, kp2, ku, met, mt))
    #  arq1.close()

    # string = monta_latex(wil, accVotingPool, accVotingPgsc, accLCAB, accLCAP, accOLAB, accOLAP, accRankB, accRankP,
    #                     accKEB, accKEP,
    #                     accKUB, accKUP, accMetaB, accMetaP)
    # arq3.write(string)
    # arq3.close()

    monta_resultados(arq2, accVotingPgsc, accVotingPool, accLCAP, accLCAB, accOLAP, accOLAB, accRankP, accRankB, accKEP,
                     accKEB, accKUP, accKUB, accMetaP, accMetaB, accVotingPgsc2, accLCAP2, accOLAP2, accRankP2, accKEP2,
                     accKUP2, accMetaP2)
    arq2.close()


def des_method(poolPgsc, pool, calibratedPool, calibratedPgsc, X_valida, y_valida, X_test, y_test, poolPgsc2=None,
               calibratedPgsc2=None):
    global accKUB, accKEB, accOLAB, accKUP, accKEP, accOLAP, accLCAB, accLCAP, accRankB, accRankP, accMetaB, accMetaP, accVotingPool, accVotingPgsc
    if poolPgsc2:
        Pgsc_voting2 = EnsembleVoteClassifier(clfs=poolPgsc2, voting='hard', refit=False)
        Pgsc_voting2 = Pgsc_voting2.fit(X_valida, y_valida)
        PgscV2 = Pgsc_voting2.score(X_test, y_test)
        metdp2 = METADES(calibratedPgsc2)
        lcap2 = LCA(poolPgsc2)
        rankp2 = Rank(poolPgsc2)
        knorauP2 = KNORAU(poolPgsc2)
        kneP2 = KNORAE(poolPgsc2)
        olaP2 = OLA(poolPgsc2)

        metdp2.fit(X_valida, y_valida)
        lcap2.fit(X_valida, y_valida)
        rankp2.fit(X_valida, y_valida)
        knorauP2.fit(X_valida, y_valida)
        kneP2.fit(X_valida, y_valida)
        olaP2.fit(X_valida, y_valida)
        accMetaP2.append(metdp2.score(X_test, y_test))
        accLCAP2.append(lcap2.score(X_test, y_test))
        accRankP2.append(rankp2.score(X_test, y_test))
        accKUP2.append(knorauP2.score(X_test, y_test))
        accKEP2.append(kneP2.score(X_test, y_test))
        accOLAP2.append(olaP2.score(X_test, y_test))
        accVotingPgsc2.append(PgscV2)

    Pgsc_voting = EnsembleVoteClassifier(clfs=poolPgsc, voting='hard', refit=False)
    Pgsc_voting = Pgsc_voting.fit(X_valida, y_valida)
    PgscV = Pgsc_voting.score(X_test, y_test)

    Pool_voting = EnsembleVoteClassifier(clfs=pool, voting='hard', refit=False)
    Pool_voting = Pool_voting.fit(X_valida, y_valida)
    PollV = Pool_voting.score(X_test, y_test)

    metdb = METADES(calibratedPool)
    metdp = METADES(calibratedPgsc)

    lcab = LCA(pool)
    lcap = LCA(poolPgsc)
    rankb = Rank(pool)
    rankp = Rank(poolPgsc)
    knorauB = KNORAU(pool)
    knorauP = KNORAU(poolPgsc)
    kneB = KNORAE(pool)
    kneP = KNORAE(poolPgsc)
    olaB = OLA(pool)
    olaP = OLA(poolPgsc)

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

    # print(lcap.score(X_test, y_test))
    # exit(0)
    accMetaB.append(metdb.score(X_test, y_test))
    accMetaP.append(metdp.score(X_test, y_test))
    accLCAB.append(lcab.score(X_test, y_test))
    accLCAP.append(lcap.score(X_test, y_test))
    accRankB.append(rankb.score(X_test, y_test))
    accRankP.append(rankp.score(X_test, y_test))
    accKUB.append(knorauB.score(X_test, y_test))
    accKUP.append(knorauP.score(X_test, y_test))
    accKEB.append(kneB.score(X_test, y_test))
    accKEP.append(kneP.score(X_test, y_test))
    accOLAB.append(olaB.score(X_test, y_test))
    accOLAP.append(olaP.score(X_test, y_test))

    accVotingPool.append(PollV)
    accVotingPgsc.append(PgscV)

    if poolPgsc2:
        return accVotingPgsc, accVotingPgsc2, accVotingPool, accLCAP, accLCAP2, accLCAB, accOLAP, accOLAP2, accOLAB, accRankP, accRankP2, accRankB, accKEP,\
               accKEP2, accKEB, accKUP, accKUP2, accKUB, accMetaP, accMetaP2, accMetaB
    else:
        return accVotingPgsc, accVotingPool, accLCAP, accLCAB, accOLAP, accOLAB, accRankP, accRankB, accKEP, accKEB, accKUP, accKUB, accMetaP, accMetaB


def Selecao(nome_base, local, cpx_caminho, bags_ga, nome_arq, repeticao, metodo, classifier, bags_ga2):
    global accKUB, accKEB,accOLAB ,accSaccKUB ,accKEB ,accOLAB,accSBB,accKUP ,accKEP,accOLAP,accSBP ,accLCAB ,accLCAP ,accRankB ,accRankP ,accMetaB ,accMetaP,\
        accVotingPool, accVotingPgsc,accKUP2,accKEP2 ,accOLAP2,accSBP2 ,accLCAP2 ,accRankP2,accMetaP2,accVotingPgsc2
    '''

    @param nome_base:
    @param local:
    @param cpx_caminho:
    @param bags_ga:
    @param nome_arq:
    @param repeticao:
    @param metodo:
    @param classifier: tipo de classificador
    @param bags_ga2: se tem mais de um experimento meu para comparar
    @return:
    '''
    arq_dataset = local + "Dataset/" + nome_base + ".arff"
    arq_arff = Marff.abre_arff(arq_dataset)
    X, y, _ = Marff.retorna_instacias(arq_arff)

    for j in range(1, repeticao):
        print(len(accLCAP), nome_arq, nome_base, metodo)
        pool = []
        poolPgsc = []
        poolPgsc2 = []  # rodar mais de um experimento
        calibratedPool = []
        calibratedPgsc = []
        calibratedPgsc2 = []

        bags2 = Cpx.open_bag(cpx_caminho + str(j) + "/", nome_base + bags_ga)
        if bags_ga2:
            bags3 = Cpx.open_bag(cpx_caminho + str(j) + "/", nome_base + bags_ga2)
        teste, validacao = Cpx.open_test_vali(local, nome_base, j)
        train = Cpx.open_training(local, nome_base, str(j))

        X_train, y_train = Cpx.biuld_x_y(train, X, y)
        X_test, y_test = Cpx.biuld_x_y(teste, X, y)
        X_valida, y_valida = Cpx.biuld_x_y(validacao, X, y)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_valida = np.array(X_valida)

        scaler = StandardScaler()
        X_valida = scaler.fit_transform(X_valida)
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)

        if metodo == "rf":
            ensemble = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=0)
            ensemble.fit(X_train, y_train)
            pool = ensemble.estimators_
            if classifier == "perc":
                for i in range(100):
                    X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
                    X_bag2 = scaler.transform(X_bag2)
                    percP = novo_perceptron.PPerceptron(n_jobs=4, tol=1.0)
                    poolPgsc.append(percP.fit(X_bag2, y_bags2))
                    if bags_ga2:
                        X_bag3, y_bags3 = Cpx.biuld_x_y(bags3['inst'][i], X, y)
                        X_bag3 = scaler.transform(X_bag3)
                        percP1 = novo_perceptron.PPerceptron(n_jobs=4, tol=1.0)
                        poolPgsc2.append(percP1.fit(X_bag3, y_bags3))

            else:
                for i in range(100):
                    X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
                    X_bag2 = scaler.transform(X_bag2)
                    tree = DecisionTreeClassifier()
                    poolPgsc.append(tree.fit(X_bag2, y_bags2))
                    if bags_ga2:
                        X_bag3, y_bags3 = Cpx.biuld_x_y(bags3['inst'][i], X, y)
                        X_bag3 = scaler.transform(X_bag3)
                        tree2 = DecisionTreeClassifier()
                        poolPgsc2.append(tree2.fit(X_bag3, y_bags3))

        elif metodo == "ada":
            if classifier == "perc":
                ensemble = AdaBoostClassifier(novo_perceptron.PPerceptron(n_jobs=4, tol=1.0), n_estimators=100,
                                              algorithm='SAMME.R')
                ensemble.fit(X_train, y_train)
                pool = ensemble.estimators_
                for i in range(100):
                    X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
                    X_bag2 = scaler.transform(X_bag2)
                    percP = novo_perceptron.PPerceptron(n_jobs=4, tol=1.0)
                    poolPgsc.append(percP.fit(X_bag2, y_bags2))
                    if bags_ga2:
                        X_bag3, y_bags3 = Cpx.biuld_x_y(bags3['inst'][i], X, y)
                        X_bag3 = scaler.transform(X_bag3)
                        percP1 = novo_perceptron.PPerceptron(n_jobs=4, tol=1.0)
                        poolPgsc2.append(percP1.fit(X_bag3, y_bags3))

            else:
                ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100,
                                         algorithm='SAMME.R')
                ada.fit(X_train, y_train)
                pool = ada.estimators_
                for i in range(100):
                    X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
                    X_bag2 = scaler.transform(X_bag2)
                    tree = DecisionTreeClassifier()
                    poolPgsc.append(tree.fit(X_bag2, y_bags2))
                    if bags_ga2:
                        X_bag3, y_bags3 = Cpx.biuld_x_y(bags3['inst'][i], X, y)
                        X_bag3 = scaler.transform(X_bag3)
                        tree2 = DecisionTreeClassifier()
                        poolPgsc2.append(tree2.fit(X_bag3, y_bags3))


        elif metodo == "bagging":
            bags = Cpx.open_bag(local + "Bags/" + str(j) + "/", nome_base)
            if classifier == "perc":
                for i in range(100):
                    X_bag, y_bag = Cpx.biuld_x_y(bags['inst'][i], X, y)
                    X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
                    X_bag = scaler.transform(X_bag)
                    X_bag2 = scaler.transform(X_bag2)
                    percB = perc(n_jobs=4, tol=1.0)
                    percP = perc(n_jobs=4, tol=1.0)
                    pool.append(percB.fit(X_bag, y_bag))
                    poolPgsc.append(percP.fit(X_bag2, y_bags2))
                    if bags_ga2:
                        X_bag3, y_bags3 = Cpx.biuld_x_y(bags3['inst'][i], X, y)
                        X_bag3 = scaler.transform(X_bag3)
                        percP1 = novo_perceptron.PPerceptron(n_jobs=4, tol=1.0)
                        poolPgsc2.append(percP1.fit(X_bag3, y_bags3))

            else:
                for i in range(100):
                    X_bag, y_bag = Cpx.biuld_x_y(bags['inst'][i], X, y)
                    X_bag2, y_bags2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
                    X_bag = scaler.transform(X_bag)
                    X_bag2 = scaler.transform(X_bag2)
                    treeB = DecisionTreeClassifier()
                    treeP = DecisionTreeClassifier()
                    pool.append(treeB.fit(X_bag, y_bag))
                    poolPgsc.append(treeP.fit(X_bag2, y_bags2))
                    if bags_ga2:
                        X_bag3, y_bags3 = Cpx.biuld_x_y(bags3['inst'][i], X, y)
                        X_bag3 = scaler.transform(X_bag3)
                        tree2 = DecisionTreeClassifier()
                        poolPgsc2.append(tree2.fit(X_bag3, y_bags3))

        for clf in pool:
            calibrated = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
            calibrated.fit(X_valida, y_valida)
            calibratedPool.append(calibrated)

        for clf in poolPgsc:
            calibrated = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
            calibrated.fit(X_valida, y_valida)
            calibratedPgsc.append(calibrated)

        if bags_ga2:
            for clf in poolPgsc2:
                calibrated = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
                calibrated.fit(X_valida, y_valida)
                calibratedPgsc2.append(calibrated)
        if bags_ga2:
            accVotingPgsc, accVotingPgsc2, accVotingPool, accLCAP, accLCAP2, accLCAB, accOLAP, accOLAP2, accOLAB, accRankP, accRankP2, accRankB, accKEP,accKEP2, accKEB, accKUP, accKUP2, accKUB, accMetaP, accMetaP2, accMetaB=des_method(poolPgsc, pool, calibratedPool, calibratedPgsc, X_valida, y_valida, X_test, y_test, poolPgsc2,
               calibratedPgsc2)
        else:
            accVotingPgsc, accVotingPool, accLCAP, accLCAB, accOLAP, accOLAB, accRankP, accRankB, accKEP, accKEB, accKUP, accKUB, accMetaP, accMetaB = \
                des_method(poolPgsc, pool, calibratedPool, calibratedPgsc, X_valida, y_valida, X_test, y_test,
                           poolPgsc2=None, calibratedPgsc2=None)

    arquivos(nome_arq, accVotingPgsc, accVotingPool, accLCAP, accLCAB, accOLAP, accOLAB, accRankP, accRankB, accKEP,
             accKEB, accKUP, accKUB, accMetaP, accMetaB, accVotingPgsc2, accLCAP2, accOLAP2, accRankP2, accKEP2,
             accKUP2, accMetaP2)
    cria_arquivo_individual(nome_arq, metodo, accVotingPgsc, accVotingPool, accLCAP,
                            accLCAB, accOLAP, accOLAB, accRankP, accRankB, accKEP, accKEB, accKUP, accKUB, accMetaP,
                            accMetaB, accVotingPgsc2, accLCAP2, accOLAP2, accRankP2, accKEP2, accKUP2, accMetaP2)

    accKUB = []
    accKEB = []
    accOLAB = []
    accSaccKUB = []
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

    accVotingPool = []
    accVotingPgsc = []
    accKUP2 = []
    accKEP2 = []
    accOLAP2 = []
    accSBP2 = []

    accLCAP2 = []
    accRankP2 = []
    accMetaP2 = []

    accVotingPgsc2 = []


# nome_base="Segmentation"
# selecao_ada(nome_base,local, cpx_caminho,bags_ga,nome_arq,repeticao=21)
# Selecao(nome_base,local, cpx_caminho,bags_ga,nome_arq,repeticao=21, metodo="bagging", classifier="tree")
# Selecao(nome_base,local, cpx_caminho,bags_ga,nome_arq,repeticao=21, metodo="ada", classifier="tree")
# Selecao(nome_base,local, cpx_caminho,bags_ga,nome_arq,repeticao=21, metodo="rf", classifier="tree")
# nome_arq="disttree"
# bags_ga="19maxdistancetree"
Selecao(nome_base, local, cpx_caminho, bags_ga, nome_arq, repeticao=21, metodo="bagging", classifier="tree", bags_ga2=bags_ga2)
Selecao(nome_base, local, cpx_caminho, bags_ga, nome_arq, repeticao=21, metodo="ada", classifier="tree", bags_ga2=bags_ga2)
Selecao(nome_base, local, cpx_caminho, bags_ga, nome_arq, repeticao=21, metodo="rf", classifier="tree", bags_ga2=bags_ga2)
