import subprocess, numpy

#caminho = "/media/marcos/Data/Tese/Bases/Teste/1/TesteWine1.arff"
dcol = "/media/marcos/Data/Tese/dcol/DCoL-v1.1/Source/dcol ./dcol"


def retorna_complexidade(caminho, complexidades, num_classes=2, media=False):
    if num_classes > 2:
        d = " -d "
    else:
        d = ' '
    #print(dcol + " -i " + caminho + ' -o /home/marcos/Área\ de\ Trabalho/c.txt' +d+ complexidades)
    proc = subprocess.Popen([dcol + " -i " + caminho + ' -o /home/marcos/Área\ de\ Trabalho/c.txt' +d+ complexidades],
                            stdout=subprocess.PIPE, shell=True)

    (cont_arq, err) = proc.communicate()

    cont_arq = (cont_arq.decode("utf-8"))
    cont_arq = cont_arq.split()
    F1 = []
    N2 = []
    N4 = []

    for i in cont_arq:
        if(i.find('F1-') != -1):
            x = i.replace('F1-', "")
            x = float(x)
            F1.append(x)
        if (i.find('N2-') != -1):
            x = i.replace('N2-', "")
            x = float(x)
            N2.append(x)
        if (i.find('N4-') != -1):
            x = i.replace('N4-', "")
            x = float(x)
            N4.append(x)

    if num_classes > 2 and media == True:
        return round(numpy.average(F1),6) if F1 else "Erro F1", round(numpy.average(N2),6) if N2 else "Erro N2",
        numpy.average(N4) if N4 else "Erro N4"
    else:
       # print(F1)
        return round(F1[0],6) if F1 else "Erro F1", round(N2[0],6) if N2 else "Erro N2", round(N4[0],6) if N4 else "Erro N4"



#F1, N2, *_ = retorna_complexidade(caminho, num_classes=3, complexidades="-F 1 -N 2", media=True)
#print(N2)
#print(F1)