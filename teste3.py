import numpy
import csv, sys
from matplotlib import pyplot as plot
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.ticker import NullFormatter  # useful for `logit` scale

nome_base = 'Wine'
nb_plots = 20
nb_plots_per_page = 10


def dsoc():
    global AG, bagging
    with open('/home/marcos/Documents/Tese/dsoc/dsoc' + nome_base + '.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        result = []

        for linha in spamreader:
            result.append((linha))

        AG = result[0][1:]
        bagging = result[1][1:]
        # labels = [str(w) for w in ag]


# exit(0)
def resultado_dsoc(f):
    t = "Dsoc\n" \
        "BAG: " + bagging[f] + "\n" + "AG: " + AG[f]

    plot.text(0.2, 1.5, s=t, bbox=dict(alpha=0.5))

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

def gera_grafico():
    # pdf_pages = PdfPages('Disp'+nome_base+'.pdf')
    # Generate the pages

    # nb_pages = int(numpy.ceil(nb_plots / float(nb_plots_per_page)))
    grid_size = (4, 2)
    j=0
    r=0
    fig = plt.figure()
    plotss = []
    iii=4
    kkk=4
    lll=10
    for i in range(1, 2):
        data1, data2, data3, data4, data5, data6=arquivo(i)
        axis1 = fig.add_subplot(iii,kkk,lll)
        axis1.plot(range(10))
        plotss.append(axis1)
        #lll+=1

    fig.savefig(nome_base + str(i) + '.png')

       #  if (i % 2 == 0):
       #      yy = 1
       #  else:
       #      yy = 0
       #  # if j % nb_plots_per_page == 0:
       #  #     print(j % nb_plots_per_page)
       #  #     print('\n')
       #  #     plot.figure(1)
       #
       #  plot.subplot2grid(grid_size, (r, yy))
       #  print(grid_size, r, yy)
       #  # plot.axis((0.06,0.2,0,3.5))
       #  plot.title(nome_base + " " + str(i))
       #  bag = plot.scatter(x=data2, y=data1, alpha=0.6, marker="v")
       #  dist = plot.scatter(x=data4, y=data3, alpha=0.6, marker='+')
       #  ag = plot.scatter(x=data6, y=data5, alpha=0.6)
       #  plot.xlabel("N2")
       #  plot.ylabel("F1")
       #  plt.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "])
       #  # print(mf1)
       #  # resultado_dsoc(f)
       #
       #
       #  # Close the page if needed
       # # if (j + 1) % nb_plots_per_page == 0 or (j + 1) == nb_plots:
       #      # 1if(r<4):
       #      #plot.tight_layout()
       #     # j = 0
       #
       #      # pdf_pages.savefig(fig)
       #  if(i==4 or i==8 or i==12 or i==16 or i==20):
       #      plot.savefig(nome_base + str(i) + '.png')
       #  if (i == 4):
       #      r = 0
       #
       #
       #  if (i % 2 == 0):
       #      r = r + 1
       #
       #      # pdf_pages.close()


gera_grafico()
# gera_png()
#
# for i in range(1,21):
# #arq1 = open('/media/marcos/Data/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
#     arq2 = open('/media/marcos/Data/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
#     arq1 = open('/media/marcos/Data/Tese/ComplexidadeBags/' + nome_base + '/' + nome_base + '_medias.txt', 'r')
#     arq3 = open('/media/marcos/Data/Tese/ComplexidadeAG/' + nome_base + '/' + nome_base + '_medias.txt', 'r')
