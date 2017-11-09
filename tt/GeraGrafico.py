
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import csv, sys




nome_base='Yonosphere'
#nome_base='Wine'
print(nome_base)
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
            print(b[1])
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
                AG=linha[:]
            if (linha[0] == nome_base + "B"):
                bagging=linha[:]
    AG = AG[1:]
    bagging = bagging[1:]
    #print(bagging)

def resultado_dsoc(f):
    if nome_base=='Banana':
        loc=4
    else:
        loc=1

    t = "Dsoc\n" \
        "BAG: " + bagging[f]+ "\nGA: " + AG[f]
    at = AnchoredText(t,
                      prop=dict(size=8), frameon=True,
                      loc=loc,
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
    fig = plt.figure(figsize=(8,13),dpi=150)
    #fig.subplots()
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    #fig = plt.figure(figsize=(23, 18), dpi=150)
    #fig.subplots_adjust(hspace=0.3, wspace=0.2, )
    for i in range(1, 9):

        ax = fig.add_subplot(4, 2, i, )
        data2, data1, data4, data3, data6, data5 = arquivo(i)
        bag=ax.scatter(data1, data2, alpha=0.6, marker="v")
        dist=ax.scatter(data3, data4, alpha=0.6, marker="+")
        ag=ax.scatter(data5, data6, alpha=0.6)
        if (nome_base == 'Yonosphere'):
            ax.set_title('Ionosphere' + str(i))
        else:
            ax.set_title(nome_base + str(i))
        ax.set_xlabel("N2")
        ax.set_ylabel("F1")
        #if(i==1):
            #ax.legend((bag, dist, ag), ("BAG. ", "KNhood. ", "G.A. "),loc=2,bbox_to_anchor=(-0.5,1), prop={'size':7})
        at = resultado_dsoc(i)
        ax.add_artist(at)

        j = i + 1

    if (nome_base == 'Yonosphere'):

        fig.savefig("Graficos/Ionosphere1.png")
    else:
        fig.savefig("Graficos/" + nome_base  +"1.png")

    fig = plt.figure(figsize=(8, 13), dpi=150)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)


    for i in range(1, 9):

        ax = fig.add_subplot(4, 2, i)

        data2, data1, data4, data3, data6, data5 = arquivo(j)
        ax.scatter(data1, data2, alpha=0.6, marker="v")
        ax.scatter(data3, data4, alpha=0.6, marker="+")
        ax.scatter(data5, data6, alpha=0.6)
        if (nome_base == 'Yonosphere'):
            ax.set_title('Ionosphere' + str(j))
        else:
            ax.set_title(nome_base + str(j))
        ax.set_xlabel("N2")
        ax.set_ylabel("F1")
        #if (j == 7):
            #ax.legend((bag, dist, ag), ("BAG. ", "KNhood. ", "G.A. "), loc=2, bbox_to_anchor=(-0.5, 1),
                     # prop={'size': 7})
        at=resultado_dsoc(j)
        ax.add_artist(at)
        #print(j)
        j+=1

        #ax.text(0.5, 0.5, str((2, 3, i)),
               #fontsize=18, ha='center')
    if(nome_base=='Yonosphere'):

         fig.savefig("Graficos/Ionosphere2.png")
    else:
        fig.savefig("Graficos/" + nome_base + "2.png")

    fig = plt.figure(figsize=(9, 6.8), dpi=150)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    for i in range(1, 5):

        ax = fig.add_subplot(2, 2, i)
        data2, data1, data4, data3, data6, data5 = arquivo(j)
        ax.scatter(data1, data2, alpha=0.6, marker="v")
        ax.scatter(data3, data4, alpha=0.6, marker="+")
        ax.scatter(data5, data6, alpha=0.6)
        if(nome_base=='Yonosphere'):
            ax.set_title('Ionosphere' + str(j))
        else:
            ax.set_title(nome_base + str(j))
       # if (j == 13):
           # ax.legend((bag, dist, ag), ("BAG. ", "KNhood. ", "G.A. "), loc=2, bbox_to_anchor=(-0.5, 1),
                #      prop={'size': 7})
        ax.set_xlabel("N2")
        ax.set_ylabel("F1")
        at = resultado_dsoc(j)
        ax.add_artist(at)
        j+=1
        #ax.text(0.5, 0.5, str((2, 3, i)),
               #fontsize=18, ha='center')
    if (nome_base == 'Yonosphere'):

        fig.savefig("Graficos/Ionosphere3.png")
    else:
        fig.savefig("Graficos/" + nome_base + "3.png")
    exit(0)
    fig = plt.figure(figsize=(15, 8), dpi=100)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, 3):

        ax = fig.add_subplot(2, 3, i)
        data2, data1, data4, data3, data6, data5 = arquivo(j)
        ax.scatter(data1, data2, alpha=0.6, marker="v")
        ax.scatter(data3, data4, alpha=0.6, marker="+")
        ax.scatter(data5, data6, alpha=0.6)
        ax.set_title(nome_base + str(j))
        #ax.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "],loc=2)
        if (j == 19):
            ax.legend((bag, dist, ag), ("BAG. ", "KNhood. ", "G.A. "), loc=2, bbox_to_anchor=(-0.5, 1),
                      prop={'size': 7})
        ax.set_xlabel("N2")
        ax.set_ylabel("F1")
        at = resultado_dsoc(j)
        ax.add_artist(at)
        j+=1
        #ax.text(0.5, 0.5, str((2, 3, i)),
               #fontsize=18, ha='center')
    fig.savefig("Graficos/"+nome_base + "4.png")


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
