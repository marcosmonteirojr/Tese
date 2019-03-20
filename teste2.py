import Cpx, Marff,os,sys
#nome_base="Banana"
local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
caminho_base = "/media/marcos/Data/Tese/Bases2/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
nome_base='CTG'

for j in range(1,21):
    arq = open(cpx_caminho+str(j)+'/'+nome_base+'20andre.csv', 'r')
    arq2 = open(cpx_caminho + str(j) + '/' + nome_base + '20andretemp.csv', 'w')
   # x=arq.readlines()
    #print(x[-1])
    #exit(0)
    cont=0
    for i in arq:

     #   print(len(i))
        if len(i) != 1:
           if cont == 99:
                arq2.write(i[:-1])
           else:
                arq2.write(i)
        cont = cont + 1
        print(cont)
    if cont<99:
        arq2.write('\n')
    os.system("mv "+cpx_caminho + str(j) + '/' + nome_base + '20andretemp.csv '+cpx_caminho+str(j)+'/'+nome_base+'20andre.csv')
    #os.system("gzip -d /home/marcos/Documentos/Bags/"+ str(j) + '/'+nome_base)
