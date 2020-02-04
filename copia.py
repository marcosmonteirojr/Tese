import os,sys
caminho1='/media/marcos/OS/Users/marca/Documents/Minhas-Imagens/xiaomi/Camera/'
caminho2=' /media/marcos/OS/Users/marca/Documents/Minhas-Imagens/Fotos-videosMiguel/xiaomi/2019/'
caminho3=' /media/marcos/OS/Users/marca/Documents/Minhas-Imagens/Fotos-videosMiguel/xiaomi/2020/'
#nome_base=sys.argv[1]
#nome_base="ILPD"

# for i in range(1,13):
#     if (os.path.exists(caminho2+"/"+str(i)) == False):
#         os.system("mkdir -p " + caminho2+"/"+str(i))
#
x=open(caminho1+'testev.txt')
#for i in range(4,13):
#    os.system("mv -n -u" +caminho2 +str(i)+"/*.mp4 "+caminho2+str(i)+"-v/")
# for i in x:
#
# #

for i in x:

    if i[6:10] =="2019":
       # print(i[10:12])
        if i[10:12]=="04":
             os.system("mv  "+ caminho1 +i[2:-1]+" " +caminho2+"4-v/")
        if i[10:12] == "05":
             os.system("mv  " + caminho1 + i[2:-1] +" " + caminho2 + "5-v/" )
        if i[10:12] == "06":
            os.system("mv  -v " + caminho1 + i[2:-1] +" " + caminho2 + "6-v/" )
        if i[10:12] == "07":
            os.system("mv  -v " + caminho1 + i[2:-1] +" " + caminho2 + "7-v/" )
        if i[10:12] == "08":
            os.system("mv  -v " + caminho1 + i[2:-1] +" " + caminho2 + "8-v/" )
        if i[10:12] == "09":
            os.system("mv   -v " + caminho1 + i[2:-1] +" " + caminho2 + "9-v/")
        if i[10:12] == "10":
            os.system("mv   -v " + caminho1 + i[2:-1] + " " +caminho2 + "10-v/")
        if i[10:12] == "11":
            os.system("mv   -v " + caminho1 + i[2:-1] + " " +caminho2 + "11-v/")
        if i[10:12] == "12":
            os.system("mv   -v " + caminho1 + i[2:-1] + " " +caminho2 + "12-v/")

    if i[6:10] == "2020":
        if i[10:12] == "01":
            print(2020)
            os.system("mv   -v " + caminho1 + i[2:-1] + " " + caminho3 + "1-v/" )
        if i[10:12] == "05":
            os.system("mv   -v " + caminho1 + i[2:] + " " + caminho2 + "05/" + i[2:])
        if i[10:12] == "06":
            os.system("mv   -v " + caminho1 + i[2:] + " " + caminho2 + "06/" + i[2:])
        if i[10:12] == "07":
            os.system("mv   -v " + caminho1 + i[2:] + " " + caminho2 + "07/" + i[2:])
        if i[10:12] == "08":
            os.system("mv   -v " + caminho1 + i[2:] + " " + caminho2 + "08/" + i[2:])
        if i[10:12] == "09":
            os.system("mv   -v " + caminho1 + i[2:] + " " + caminho2 + "09/" + i[2:])
        if i[10:12] == "10":
            os.system("mv   -v " + caminho1 + i[2:] + " " + caminho2 + "10/" + i[2:])
        if i[10:12] == "11":
            os.system("mv   -v " + caminho1 + i[2:] + " " + caminho2 + "11/" + i[2:])
        if i[10:12] == "12":
            os.system("mv   -v " + caminho1 + i[2:] + " " + caminho2 + "12/" + i[2:])
        #exit(0)
#         c=i.split("\n")[0]
#         if c[-1]=="4":
#             os.system("mv -n -u" + caminho1 + c + caminho3 )
#         #exit(0)
#         print(c[4:6])
#         if c[4:6]=="01":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "1/")
#         elif c[4:6] == "02":
#             os.system("mv -n -u " + caminho1 + c + caminho2 + "2/")
#         if c[4:6]=="03":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "3/")
#         if c[4:6]=="04":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "4/")
#         if c[4:6]=="05":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "5/")
#         if c[4:6]=="06":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "6/")
#         if c[4:6]=="07":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "7/")
#         if c[4:6]=="08":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "8/")
#         if c[4:6]=="09":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "9/")
#         if c[4:6]=="10":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "10/")
#         if c[4:6]=="11":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "11/")
#         if c[4:6]=="12":
#             os.system("mv -n -u "+caminho1+c+caminho2+ "12/")
# #print(c[0])
# #c=x.split("\n")
#
# exit(0)
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
# geracao=0
# for i in range(1,21):
#     repeticao=i
#     altera_arquivo_marcelo()


#for i in range(1,21):
    # for j in range(1,30):
    #     if j<28:
    #         os.system("rm /home/projeto/Marcos/GA5/"+str(i)+"Banana"+str(j)+".indx")
 #           os.system("mv /media/marcos/Data/Tese/Bases3/Bags/" + str(i) + "/*gif* "  "/media/marcos/Data/Tese/Bases3/aq_p_gif/" )
            # os.system("mv /media/marcos/Data/Tese/Bases3/Bags/" + str(
            # i) + "/P219distdiverlinear_teste_parada_dist.csv "  "/media/marcos/Data/Tese/Bases3/Bags/" + str(
            # i) + "/P220distdiverlinear_teste_parada_dist.csv")