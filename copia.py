import os,sys
caminho1='/media/marcos/Data/Tese//'
caminho2='/media/marcos/Data/Tese/Bases2/Validacao/'
nome_base=sys.argv[1]
#nome_base="ILPD"
#for i in range(1,21):
   # if (os.path.exists(caminho2+"/"+str(i)) == False):
        #os.system("mkdir -p " + caminho2+"/"+str(i))
#for i in range(1,21):
    #os.system("cp -r "+ caminho1+str(i)+"/validation/* " +caminho2+str(i)+"/")
    #os.system("rm -rf "+caminho1+"/" +str(i) + "/*.indx" )
    #os.system("gunzip "+caminho1+str(i)+"/"+nome_base+".idx.gz")
    #os.system("cp -r "+caminho1+str(i)+"/"+nome_base+".idx " +caminho1+str(i)+"/"+nome_base+"0.indx")
    #os.system("cp -r " + caminho1 + str(i) + "/Ionosphere.idx " + caminho1 + str(i) + "/Yonosphere.idx")

def altera_arquivo_marcelo():
    '''
    da nome aos bags nesse caso 1 a 100
    :return:
    '''
    global repeticao, nome_base, geracao
    arq = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base +str(geracao)+ ".indx")
    arqtemp = open("/media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + ".indxTemp", 'w')
    cont=1
    for i in arq:
        texto=i
        q=texto.split(" ")
        #print(q)
        q.insert(0,str(cont))
        #print(q)
        for j in q:
           # print(j)
            if(j!=q[-1]):
                arqtemp.write(j)
                arqtemp.write(" ")
            else:
                arqtemp.write(j)
                #arqtemp.write('\n')
        cont+=1
    arq.close()
    arqtemp.close()
    os.system("cp -r /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base + ".indxTemp /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base +str(geracao)+ ".indx")
    os.system("rm /media/marcos/Data/Tese/GA2/" + str(repeticao) + "/" + nome_base +".indxTemp")
geracao=0
for i in range(1,21):
    repeticao=i
    altera_arquivo_marcelo()


for i in range(1,21):
    for j in range(1,30):
        if j<28:
            os.system("rm /home/projeto/Marcos/GA5/"+str(i)+"Banana"+str(j)+".indx")
        if j ==29 or j==30:
            os.system("mv /home/projeto/Marcos/GA5/" + str(i) + "Banana" + str(j) + ".indx" "/home/projeto/Marcos/GA5/" + str(i) + "Banana" + str(j) + "-5.indx")