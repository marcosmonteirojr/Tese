import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import csv




nome_base='Adult'
def arquivo(i):
    data3 = []
    data4 = []
    data1 = []
    data2 = []
    data5 = []
    data6 = []
    # arq1 = open('/media/marcos/Data/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
    arq2 = open('/media/marcos/Data/Tese/ComplexidadeDistancias/' + nome_base + '/' + nome_base + str(
        i) + '/' + nome_base + '_medias.txt', 'r')
    arq1 = open('/media/marcos/Data/Tese/ComplexidadeBags/' + nome_base + '/' + nome_base + str(
        i) + '/' + nome_base + '_medias.txt', 'r')
    arq3 = open('/media/marcos/Data/Tese/ComplexidadeAg/' + nome_base + '/' + nome_base + str(
        i) + '/' + nome_base + '_medias.txt', 'r')

    for k in arq1:
        b = k.replace(', ', ' ')
        b = b.split(' ')
        if b[0] == 'F1' or b[1] == 'inf':
            None
        else:
            # print(b[1])
            data1.append(b[0])
            data2.append(b[1])
            data1 = [float(w) for w in data1]
            data2 = [float(w) for w in data2]
            # print (data1)
            # data1 = (data1[1:])
    for k in arq2:
        b = k.replace(', ', ' ')
        b = b.split(' ')
        if b[0] == 'F1' or b[1] == 'inf':
            None
        else:
            # print(b[1])
            data3.append(b[0])
            data4.append(b[1])
            data3 = [float(w) for w in data3]
            data4 = [float(w) for w in data4]
    for k in arq3:
        b = k.replace(', ', ' ')
        b = b.split(' ')
        if b[0] == 'F1' or b[1] == 'inf':
            None
        else:
            # print(b[1])
            data5.append(b[0])
            data6.append(b[1])
            data5 = [float(w) for w in data5]
            data6 = [float(w) for w in data6]
    return data1,data2,data3,data4,data5,data6





def geragraph(fig):

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9,
                       hspace=0.7, wspace=0.5)

    plt.show()


def dsoc(nome_base):
    global AG, bagging
    with open('/media/marcos/Data/Tese/dsoc/dsoc.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        AG=[]
        bagging=[]

        for linha in spamreader:
            if(linha[0]==nome_base+"A"):
                AG1=linha[:]
            if (linha[0] == nome_base + "B"):
                bagging=linha[:]
    AG = AG[1:]
    bagging = bagging[1:]
    print(bagging)

def resultado_dsoc(f):
    t = "Dsoc\n" \
        "BAG: " + bagging[f] + "\n" + "GA: " + AG[f]
    at = AnchoredText(t,
                      prop=dict(size=8), frameon=True,
                      loc=3,
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    return at

def Graficos_axs():
    cont = 1

    fig, axs = plt.subplots(3, 2)

    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[0, 0].scatter(data1,data2,alpha=0.6,marker="v")
    axs[0, 0].scatter(data3,data4,alpha=0.6,marker="+")
    axs[0, 0].scatter(data5,data6,alpha=0.6)
    axs[0, 0].set_title(nome_base+str(cont))
    axs[0,0].set_xlabel("N2")
    axs[0,0].set_ylabel("F1")
    cont+=1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[0, 1].scatter(data1,data2,alpha=0.6,marker="v")
    axs[0, 1].scatter(data3,data4,alpha=0.6,marker="+")
    axs[0, 1].scatter(data5,data6,alpha=0.6)
    axs[0, 1].set_title(nome_base+str(cont))
    axs[0,1].set_xlabel("N2")
    axs[0,1].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[1, 0].scatter(data1,data2,alpha=0.6,marker="v")
    axs[1, 0].scatter(data3,data4,alpha=0.6,marker="+")
    axs[1, 0].scatter(data5,data6,alpha=0.6)
    axs[1, 0].set_title(nome_base+str(cont))
    axs[1,0].set_xlabel("N2")
    axs[1,0].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[1, 1].scatter(data1,data2,alpha=0.6,marker="v")
    axs[1, 1].scatter(data3,data4,alpha=0.6,marker="+")
    axs[1, 1].scatter(data5,data6,alpha=0.6)
    axs[1, 1].set_title(nome_base+str(cont))
    axs[1,1].set_xlabel("N2")
    axs[1,1].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[2, 0].scatter(data1,data2,alpha=0.6,marker="v")
    axs[2, 0].scatter(data3,data4,alpha=0.6,marker="+")
    axs[2, 0].scatter(data5,data6,alpha=0.6)
    axs[2, 0].set_title(nome_base+str(cont))
    axs[2,0].set_xlabel("N2")
    axs[2,0].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[2, 1].scatter(data1, data2,alpha=0.6,marker="v")
    axs[2, 1].scatter(data3, data4,alpha=0.6,marker="+")
    axs[2, 1].scatter(data5, data6,alpha=0.6)
    axs[2, 1].set_title(nome_base+str(cont))
    axs[2,1].set_xlabel("N2")
    axs[2,1].set_ylabel("F1")

    geragraph(fig)
    fig, axs = plt.subplots(3, 2)
    cont+=1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[0, 0].scatter(data1, data2,alpha=0.6,marker="v")
    axs[0, 0].scatter(data3, data4,alpha=0.6,marker="+")
    axs[0, 0].scatter(data5, data6,alpha=0.6)
    axs[0, 0].set_title(nome_base+str(cont))
    axs[0,0].set_xlabel("N2")
    axs[0,0].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[0, 1].scatter(data1, data2,alpha=0.6,marker="v")
    axs[0, 1].scatter(data3, data4,alpha=0.6,marker="+")
    axs[0, 1].scatter(data5, data6,alpha=0.6)
    axs[0, 1].set_title(nome_base+str(cont))
    axs[0,1].set_xlabel("N2")
    axs[0,1].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[1, 0].scatter(data1, data2,alpha=0.6,marker="v")
    axs[1, 0].scatter(data3, data4,alpha=0.6,marker="+")
    axs[1, 0].scatter(data5, data6,alpha=0.6)
    axs[1, 0].set_title(nome_base+str(cont))
    axs[1,0].set_xlabel("N2")
    axs[1,0].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[1, 1].scatter(data1, data2,alpha=0.6,marker="v")
    axs[1, 1].scatter(data3, data4,alpha=0.6,marker="+")
    axs[1, 1].scatter(data5, data6,alpha=0.6)
    axs[1, 1].set_title(nome_base+str(cont))
    axs[1,1].set_xlabel("N2")
    axs[1,1].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[2, 0].scatter(data1, data2,alpha=0.6,marker="v")
    axs[2, 0].scatter(data3, data4,alpha=0.6,marker="+")
    axs[2, 0].scatter(data5, data6,alpha=0.6)
    axs[2, 0].set_title(nome_base+str(cont))
    axs[2,0].set_xlabel("N2")
    axs[2,0].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[2, 1].scatter(data1, data2,alpha=0.6,marker="v")
    axs[2, 1].scatter(data3, data4,alpha=0.6,marker="+")
    axs[2, 1].scatter(data5, data6,alpha=0.6)
    axs[2, 1].set_title(nome_base+str(cont))
    axs[2,1].set_xlabel("N2")
    axs[2,1].set_ylabel("F1")
    geragraph(fig)


    fig, axs = plt.subplots(3, 2)

    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[0, 0].scatter(data1, data2,alpha=0.6,marker="v")
    axs[0, 0].scatter(data3, data4,alpha=0.6,marker="+")
    axs[0, 0].scatter(data5, data6,alpha=0.6)
    axs[0, 0].set_title(nome_base+str(cont))
    axs[0,0].set_xlabel("N2")
    axs[0,0].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[0, 1].scatter(data1, data2,alpha=0.6,marker="v")
    axs[0, 1].scatter(data3, data4,alpha=0.6,marker="+")
    axs[0, 1].scatter(data5, data6,alpha=0.6)
    axs[0, 1].set_title(nome_base+str(cont))
    axs[0,1].set_xlabel("N2")
    axs[0,1].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[1, 0].scatter(data1, data2,alpha=0.6,marker="v")
    axs[1, 0].scatter(data3, data4,alpha=0.6,marker="+")
    axs[1, 0].scatter(data5, data6,alpha=0.6)
    axs[1, 0].set_title(nome_base+str(cont))
    axs[1,0].set_xlabel("N2")
    axs[1,0].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[1, 1].scatter(data1, data2,alpha=0.6,marker="v")
    axs[1, 1].scatter(data3, data4,alpha=0.6,marker="+")
    axs[1, 1].scatter(data5, data6,alpha=0.6)
    axs[1, 1].set_title(nome_base+str(cont))
    axs[1,1].set_xlabel("N2")
    axs[1,1].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[2, 0].scatter(data1, data2,alpha=0.6,marker="v")
    axs[2, 0].scatter(data3, data4,alpha=0.6,marker="+")
    axs[2, 0].scatter(data5, data6,alpha=0.6)
    axs[2, 0].set_title(nome_base+str(cont))
    axs[2,0].set_xlabel("N2")
    axs[2,0].set_ylabel("F1")
    cont += 1
    data2, data1, data4, data3, data6, data5 = arquivo(cont)
    axs[2, 1].scatter(data1, data2,alpha=0.6,marker="v")
    axs[2, 1].scatter(data3, data4,alpha=0.6,marker="+")
    axs[2, 1].scatter(data5, data6,alpha=0.6)
    axs[2, 1].set_title(nome_base+str(cont))
    axs[2,1].set_xlabel("N2")
    axs[2,1].set_ylabel("F1")
    geragraph(fig)

def Graficos():
    dsoc(nome_base)
    fig = plt.figure(figsize=(640, 800), dpi=100)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)
        data2, data1, data4, data3, data6, data5 = arquivo(i)
        bag=ax.scatter(data1, data2, alpha=0.6, marker="v")
        dist=ax.scatter(data3, data4, alpha=0.6, marker="+")
        ag=ax.scatter(data5, data6, alpha=0.6)
        ax.set_title(nome_base + str(i))
        ax.set_xlabel("N2")
        ax.set_ylabel("F1")
        ax.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "],loc=2)
        at = resultado_dsoc(i)
        ax.add_artist(at)

        j = i + 1

    plt.show()
    exit(0)
    fig = plt.figure(figsize=(640,800), dpi=100)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)


    for i in range(1, 7):

        ax = fig.add_subplot(2, 3, i)

        data2, data1, data4, data3, data6, data5 = arquivo(j)
        ax.scatter(data1, data2, alpha=0.6, marker="v")
        ax.scatter(data3, data4, alpha=0.6, marker="+")
        ax.scatter(data5, data6, alpha=0.6)
        ax.set_title(nome_base + str(j))
        ax.set_xlabel("N2")
        ax.set_ylabel("F1")
        ax.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "],loc=2)
        at=resultado_dsoc(j)
        ax.add_artist(at)
        print(j)
        j+=1

        #ax.text(0.5, 0.5, str((2, 3, i)),
               #fontsize=18, ha='center')
    plt.show()

    fig = plt.figure(figsize=(640, 800), dpi=100)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, 7):

        ax = fig.add_subplot(2, 3, i)
        data2, data1, data4, data3, data6, data5 = arquivo(j)
        ax.scatter(data1, data2, alpha=0.6, marker="v")
        ax.scatter(data3, data4, alpha=0.6, marker="+")
        ax.scatter(data5, data6, alpha=0.6)
        ax.set_title(nome_base + str(j))
        ax.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "],loc=2)
        ax.set_xlabel("N2")
        ax.set_ylabel("F1")
        at = resultado_dsoc(j)
        ax.add_artist(at)
        j+=1
        #ax.text(0.5, 0.5, str((2, 3, i)),
               #fontsize=18, ha='center')
    plt.show()
    fig = plt.figure(figsize=(640, 800), dpi=100)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, 3):

        ax = fig.add_subplot(2, 3, i)
        data2, data1, data4, data3, data6, data5 = arquivo(j)
        ax.scatter(data1, data2, alpha=0.6, marker="v")
        ax.scatter(data3, data4, alpha=0.6, marker="+")
        ax.scatter(data5, data6, alpha=0.6)
        ax.set_title(nome_base + str(j))
        ax.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "],loc=2)
        ax.set_xlabel("N2")
        ax.set_ylabel("F1")
        at = resultado_dsoc(j)
        ax.add_artist(at)
        j+=1
        #ax.text(0.5, 0.5, str((2, 3, i)),
               #fontsize=18, ha='center')
    plt.show()
# fig, axs = plt.subplots(2, 1)
#
# cont += 1
# data2, data1, data4, data3, data6, data5 = arquivo(cont)
# axs[0, 0].scatter(data1, data2)
# axs[0, 0].scatter(data3, data4,alpha=0.6,marker="+")
# axs[0, 0].scatter(data5, data6,alpha=0.6)
# axs[0, 0].set_title(nome_base+str(cont))
# axs[0,0].set_xlabel("N2")
# axs[0,0].set_ylabel("F1")
# cont += 1
# data2, data1, data4, data3, data6, data5 = arquivo(cont)
# axs[1, 0].scatter(data1, data2,alpha=0.6,marker="v")
# axs[1, 0].scatter(data3, data4,alpha=0.6,marker="+")
# axs[1, 0].scatter(data5, data6,alpha=0.6)
# axs[1, 0].set_title(nome_base+str(cont))
# axs[1,0].set_xlabel("N2")
# axs[1,0].set_ylabel("F1")
# cont += 1
# data2, data1, data4, data3, data6, data5 = arquivo(cont)
# bag=axs[0, 2].scatter(data1, data2,alpha=0.6,marker="v")
# dist=axs[0, 2].scatter(data3, data4,alpha=0.6,marker="+")
# ag=axs[0, 2].scatter(data5, data6,alpha=0.6)
# axs[0, 2].set_title(nome_base+str(cont))
# plt.xlabel("N2")
# plt.ylabel("F1")
# plt.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "])
#geragraph(fig)
Graficos()



# change whisker length