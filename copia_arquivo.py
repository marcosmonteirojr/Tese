import os, shutil
def copia_arquivos():
    for l in range(1,21):
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
    for l in range(1,21):
         #os.system ("cp -r /media/marcos/Data/Tese/AG/"+str(l)+"/"+str(l)+"-finais /media/marcos/Marcos/Tese/GA1Distancia" )
         #for i in range(1,30):
             os.system ("rm -r /media/marcos/Data/Tese/Bases3/Bags/" + str(l)+"/P2*" )

#limpa_pasta()

#for i in range(1,21):
#    os.system("mv /media/marcos/Data/Tese/Bases3/Bags/" + str(i) + "/"+str(i)+"/ " + "/media/marcos/Data/Tese/Joaquina2005/Bases3/Bags/" + str(i) + "/")

for j in range(1,21):
     nome = open("/media/marcos/Data/Tese/Joaquina2005/Bases3/Bags/"+str(j)+"/"+str(j)+"/nome.txt", "r")
     nome2 = open("/media/marcos/Data/Tese/Joaquina2005/Bases3/Bags/"+str(j)+"/"+str(j)+"/nome2.txt", "r")
     nome = nome.read().split('\n')
     nome2 = nome2.read().split('\n')
     for i in range(len(nome)):
         os.system("mv /media/marcos/Data/Tese/Joaquina2005/Bases3/Bags/"+str(j)+"/"+nome[i] +" /media/marcos/Data/Tese/Joaquina2005/Bases3/Bags/"+str(j)+"/"+nome2[i])
#     #print(nome[i])

import os, sys, Marff, random
# caminho1='/media/marcos/Marcos/PEREIRA/Exec_no/'
# caminho2='/media/marcos/Data/Tese/Bases2/Treino'
# nome_base='Segmentation'
# #for i in range(1,21):
#  #   if (os.path.exists(caminho2+"/"+str(i)+"/Originais") == False):
#   #      os.system("mkdir -p " + caminho2+"/"+str(i)+"/Originais")
# for i in range(1,21):
#     #for j in range (1,30):
#     #os.system("cp -r " + caminho2 + str(i) + "/Segmentation.idx " + caminho2 + str(i) + "/Segmentation0.indx")
#     #os.system("cp -r "+ caminho1+str(i)+"/train/Segmentation.idx.gz " +caminho2+str(i)+"/")
#         #os.system("rm "+caminho2+"/" +str(i) + "/"+nome_base+str(j)+".indx" )
#     #for j in range (1,101):
#         os.system("mv "+caminho2+"/"+str(i)+"/Yonosphere.idx " +caminho2+"/"+str(i)+"/Ionosphere.idx")
#         os.system("mv " + caminho2 + "/" + str(i) + "/LPD.idx " + caminho2 + "/" + str(i) + "/ILPD.idx")
#         os.system("mv " + caminho2 + "/" + str(i) + "/Image.idx " + caminho2 + "/" + str(i) + "/Segmentation.idx")
#        # os.system("mv 654"+caminho2+"/"+str(i)+"/"+str(i)+"-finais/IndividuoLPD"+str(j)+'.arff ' +caminho2+"/"+str(i)+"/"+str(i)+"-finais/IndividuoILPD"+str(j)+'.arff')
#         #os.system("mv " +caminho2 + "/" + str(i) + "/"+str(i)+"-finais/IndividuosSegmentation" + str(j) + '.arff ' + caminho2 + "/" + str(i) + "/"+str(i)+"-finais/IndividuoSegmentation" + str(j) + '.arff')
#     #c=0
def arruma_cagada_andre():
    classes=[]
    nome_base='Glass'
    caminho2 = '/media/marcos/Data/Tese/Bases2/Dataset/'
    caminho1 = '/media/marcos/Data/Tese/AG/'
    arq=Marff.abre_arff(caminho2+nome_base+".arff")
    X = dict()
    X_data=dict()
    X['data'] = list()
    X_data['data'] = list()


    X_dataset, y_dataset,X_data['data']=Marff.retorna_instacias(arq)
    for i in range(len(y_dataset)):
        if y_dataset[i]==4:
            print (i)
            classes.append(i)
    for i in range(1,21):
        for j in range(1,101):
            arq2 = Marff.abre_arff(caminho1+str(i)+'/'+str(i)+'-finais/Individuo'+ nome_base+str(j)+".arff")
            _,todas_classes,*_=Marff.retorna_classes_existentes(arq2)
            #print(todas_classes)

            _,_,X['data'] = Marff.retorna_instacias(arq2)
            #print(arq2['data'])
            instancia=random.sample(classes,1)
            #print(instancia)
           # print(X_data['data'][instancia[0]])
            X['data'].append(X_data['data'][instancia[0]])
            #print(X['data'])
            Marff.cria_arff(arq2,X,todas_classes,caminho1+str(i)+'/'+str(i)+'-finais/','Individuo'+ nome_base+str(j))
            #exit(0)
def arruma_cagada_andre_indices():
    nome_base = 'Glass'
    caminho2 = '/media/marcos/Data/Tese/Bases2/Dataset/'
    caminho1 = '/media/marcos/Data/Tese/GA2/'
    arq = Marff.abre_arff(caminho2 + nome_base + ".arff")
    classes = []
    X_dataset, y_dataset, _ = Marff.retorna_instacias(arq)
    for i in range(len(y_dataset)):
        if y_dataset[i]==4:
            #print (i)
            classes.append(i)
    print(classes)
    for i in range(1,21):
        arq=open(caminho1+str(i)+"/"+nome_base+"0.indx", 'r')

        for j in arq:
            texto=j
            texto=texto[:-1].split(" ")
            instancia = random.sample(classes, 1)
            texto.append(str(instancia[0]))
            #print(texto[-1])
            arq2 = open(caminho1 + str(i) + "/" + nome_base + "0.indxT", 'a')
            for k in texto:
                #print(k)
                if (k != texto[-1]):
                    arq2.write(k)
                    arq2.write(" ")
                    print(k, texto[-1])
                else:
                    arq2.write(k)
                arq2.close()
                    #arq2.write("\n")
        exit(0)
    arq.close()

#arruma_cagada_andre_indices()
#os.system("mv /media/marcos/Marcos/fotos-videos/fotos-12-17-a-10-3-18/*.mp4 /media/marcos/Marcos/fotos-videos/videos-12-17-a-10-3-18/")
#limpa_pasta()