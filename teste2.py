import Cpx, Marff
nome_base="Banana"
local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"

arq_dataset = caminho_base + "Dataset/" + nome_base + ".arff"
arq_arff = Marff.abre_arff(arq_dataset)
X, y, _ = Marff.retorna_instacias(arq_arff)
for i in range (16,21):
    X_train, y_train, X_test, y_test, X_vali, y_vali, dic    =Cpx.routine_save_bags(local_dataset, local, nome_base,i)