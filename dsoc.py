import Marff, Cpx, csv
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

nome_base=sys.argv[1]
#bags_ga=sys.argv[2]
#print(bags_ga.split(','))
#exit(0)

#nome_base='Wine'
local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
bags_ga="20ep"
#min_score=0

#local_dataset = "/home/projeto/Marcos/Bases2/Dataset/"
#local = "/home/projeto/Marcos/Bases3/"
#caminho_base = "/home/projeto/Marcos/Bases2/"
#cpx_caminho="/home/projeto/Marcos/Bases3/Bags/"


arq_dataset = caminho_base + "Dataset/" + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
_,classes=Marff.retorna_classes_existentes(arq_arff)
X, y, data = Marff.retorna_instacias(arq_arff)



arq = open('SelecaoMedia_desvio_pgcs_ga6.csv', 'a')
arq1 = open('SelecaoWilcoxon_pgcs_ga6.csv', 'a')
arq2 = open('SelecaoTabela_pgcs_ga6.csv', 'a')
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

for j in range(1,21):

    poolBag = []
    poolPgsc = []
    calibrated_poolBag = []
    calibrated_poolP = []
    npoolBag=[]
    npoolPgsc=[]

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
   ############################################################################################
    clf_model = pds_pool4.PClassifier(max_iter=100,tol=10.0)
    poolBag = pds_pool4.PPool(base_estimator=clf_model, classes=classes, n_estimators=100)
    poolBag.fit(bags['inst'],X,y)
    poolBag.evaluate(X_valida, y_valida)
    print(poolBag.classes_)

    #clf_model = pds_pool4.PClassifier(max_iter=100, tol=10.0)
    poolBag2 = pds_pool4.PPool(base_estimator=clf_model, classes=classes, n_estimators=100)
    poolBag2.fit(bags2['inst'], X, y)
    poolBag2.evaluate(X_valida, y_valida)

    ##########################################################
    dsocb = pds_dsoc2.DSOC2(poolBag)
    doscp = pds_dsoc2.DSOC2(poolBag2)
   ############################################

    dsocb.fit(X_valida,y_valida)
    doscp.fit(X_valida,y_valida)

    accDsocp.append(doscp.score(X_test,y_test))
    accDsocb.append(dsocb.score(X_test,y_test))

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


arq1.write('{};{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{};;{};{}\n'.format(nome_base,kp,ke,kp2,ku,op,ol,sp,sb, lca,lc, rk,rkb,met,mt,vot,votc,dsc,dsoc))
arq.write('{};{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{};;{};{};{};{}\n'
.format(nome_base,100*average(accKUB),100*std(accKUB),100*average(accKUP),100*std(accKUP),
100*average(accKEB),100*std(accKEB),100*average(accKEP),100*std(accKEP),
100*average(accOLAB),100*std(accOLAB),100*average(accOLAP),100*std(accOLAP),
100*average(accSBB),100*std(accSBB),100*average(accSBP),100*std(accSBP),
100*average(accLCAB),100*std(accLCAB),100*average(accLCAP), 100*std(accLCAP),
100*average(accRankB), 100*std(accRankB),100*average(accRankP),100*std(accRankP),
100*average(accMetaB),100*std(accMetaB),100*average(accMetaP),100*std(accMetaP),
100*average(accVotingBag),100*std(accVotingBag),100*average(accVotingPgsc),100*std(accVotingPgsc),
100*average(accDsocb),100*std(accDsocb),100*average(accDsocp),100*std(accDsocp)))

arq2.write('{};{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({});;{} ({});{} ({})\n'.format(nome_base,
round(100*average(accKUB),1),round(100*std(accKUB),1),round(100*average(accKUP),1), round(100*std(accKUP),1),
round(100*average(accKEB),1),round(100*std(accKEB),1),round(100*average(accKEP),1),round(100*std(accKEP),1),
round(100*average(accOLAB),1),round(100*std(accOLAB),1),round(100*average(accOLAP),1),round(100*std(accOLAP),1),
round(100*average(accSBB),1),round(100*std(accSBB),1),round(100*average(accSBP),1),round(100*std(accSBP),1),
round(100*average(accLCAB),1),round(100*std(accLCAB),1), round(100*average(accLCAP),1),round(100*std(accLCAP),1),
round(100*average(accRankB),1),round(100*std(accRankB),1),round(100*average(accRankP),1),round(100*std(accRankP),1),
round(100*average(accMetaB),1),round(100*std(accMetaB),1),round(100*average(accMetaP),1),round(100*std(accMetaP),1),
round(100*average(accVotingBag),1),round(100*std(accVotingBag),1),round(100*average(accVotingPgsc),1),round(100*std(accVotingPgsc),1),
round(100*average(accDsocb),1),round(100*std(accDsocb),1),round(100*average(accDsocp),1),round(100*std(accDsocp),1)))
arq1.close()
arq2.close()
arq.close()


