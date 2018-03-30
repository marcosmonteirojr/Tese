import csv, newDcol, Marff as arff
from sklearn.linear_model import perceptron

def populacao(arquivo):
    linha=[]
    arq = open(arquivo)
    for i in arq:
        linha=i[:-1].split(";")

    return linha
def CMacc(caminho_todas, caminho_teste, nome_base, repeticao, geracao, num_classes,arquivo):
    melhorAcc = open('accXcomplexS' + nome_base + str(repeticao) + '.csv','w')
    melhorAcc.write('F1;N2;Acc\n')
    linha = populacao(arquivo)
    v = caminho_teste + str(repeticao) + "/Teste" + nome_base + str(
        repeticao) + ".arff"  # e um arquivo so de validacao por repeticao
    base_teste = arff.abre_arff(v)
    X_test, y_test = arff.retorna_instacias(base_teste)
    for m in range(len(linha)):

        if(m>0):
            #print(linha[i])
            c = caminho_todas + str(repeticao) + "/" + str(geracao) + "/Individuo" + nome_base + linha[m] + ".arff"
            f1, n2, *_ = newDcol.retorna_complexidade(c, complexidades="-F 1 -N 2", num_classes=num_classes, media=False)
            base = arff.abre_arff(c)
            X, y = arff.retorna_instacias(base)
            perc = perceptron.Perceptron()
            perc.fit(X, y)
            acMoga=(perc.score(X_test,y_test))
            melhorAcc.write('{};{};{}\n'.format(f1,n2,acMoga))

