import os, shutil
def copia_arquivos():
    for l in range(2,21):
        arq=open('LogPop'+str(l)+'.csv')
        caminho_todas = "/media/marcos/Data/Tese/AG/"+str(l)+"/"
        if (os.path.exists("/media/marcos/Data/Tese/AG/"+str(l)+"/"+str(l)+"-finais_complex") == False):
            os.system("mkdir -p " + "/media/marcos/Data/Tese/AG/"+str(l)+"/"+str(l)+"-finais_complex")
            print(l)
        for i in arq:
            k=0
            linha = i[:-1].split(";")
            nome_base=linha[0]
            for j in range(len(linha)):
                if j>2:
                    k = k + 1
                    shutil.copy2(caminho_todas+'30/Individuo'+nome_base+str(linha[j])+'.arff',caminho_todas+str(l)+"-finais_complex/Individuo"+nome_base+str(k)+'.arff')
                    #print(nome_base+linha[j])

                #print(k)
def limpa_pasta():
    for l in range(1,30):
         #os.system ("cp -r /media/marcos/Data/Tese/AG/"+str(l)+"/"+str(l)+"-finais /media/marcos/Marcos/Tese/GA1Distancia" )
         for i in range(1,30):
             os.system ("rm -rf /media/marcos/Data/Tese/AG/" + str(l) + "/"+str(i))

limpa_pasta()


import os
caminho1='/media/marcos/Data/Tese/GA2/'
caminho2='/media/marcos/Data/Tese/Bases2/Validacao/'
#for i in range(1,21):
   # if (os.path.exists(caminho2+"/"+str(i)) == False):
        #os.system("mkdir -p " + caminho2+"/"+str(i))
for i in range(1,21):
    #os.system("cp -r "+ caminho1+str(i)+"/validation/* " +caminho2+str(i)+"/")
    #os.system("rm -rf "+caminho1+"/" +str(i) + "/*-info.info" )
    c=0