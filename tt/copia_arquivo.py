import os, shutil
for l in range(5,6):
    arq=open('LogPop'+str(l)+'.csv')
    caminho_todas = "/media/marcos/Data/Tese/AG/"+str(l)+"/"
    if (os.path.exists("/media/marcos/Data/Tese/AG/"+str(l)+"/"+str(l)+"-finais") == False):
        os.system("mkdir -p " + "/media/marcos/Data/Tese/AG/"+str(l)+"/"+str(l)+"-finais")
        print(l)
    for i in arq:
        k=0
        linha = i[:-1].split(";")
        nome_base=linha[0]
        for j in range(len(linha)):
            if j>2:
                k = k + 1
                shutil.copy2(caminho_todas+'30/Individuo'+nome_base+str(linha[j])+'.arff',caminho_todas+str(l)+"-finais/Individuo"+nome_base+str(k)+'.arff')
                #print(nome_base+linha[j])

                #print(k)
