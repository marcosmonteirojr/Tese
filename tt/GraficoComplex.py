import numpy
import csv
from matplotlib import pyplot as plot
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
nome_base="Adult"
nb_plots = 20
nb_plots_per_page = 4

def dsoc():
    global  AG, bagging
    with open('/home/marcos/Documents/Tese/dsoc/dsoc'+nome_base+'.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        result=[]
    
        for linha in spamreader:
            result.append((linha))
    
        AG=result[0][1:]
        bagging=result[1][1:]
    #labels = [str(w) for w in ag]


#exit(0)
def resultado_dsoc(f):
    t = "Dsoc\n" \
        "BAG: " + bagging[f] + "\n" + "AG: " + AG[f]

    plot.text(0.2,1.5,s=t, bbox=dict(alpha=0.5))


def gera_grafico():
    pdf_pages = PdfPages('Disp'+nome_base+'.pdf')
    # Generate the pages

    nb_pages = int(numpy.ceil(nb_plots / float(nb_plots_per_page)))
    grid_size = (nb_plots_per_page, 2)



    j = 0
    r=0
    for i in range(1, 21):
        data3 = []
        data4 = []
        data1 = []
        data2 = []
        data5 = []
        data6 = []
        # arq1 = open('/media/marcos/Data/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
        arq2 = open('/media/marcos/Data/Tese/ComplexidadeDist/'+nome_base+'/' + nome_base + str(i) + '/'+nome_base+'_medias.txt', 'r')
        arq1 = open('/media/marcos/Data/Tese/ComplexidadeBags/'+nome_base+'/' + nome_base + str(i)+'/' + nome_base + '_medias.txt', 'r')
        arq3 = open('/media/marcos/Data/Tese/ComplexidadeAg/'+nome_base+'/' + nome_base + str(i)+ '/' + nome_base + '_medias.txt', 'r')

        for k in arq1:
            b = k.replace(', ', ' ')
            b=b.split(' ')
            if b[0] == 'F1'or b[1]=='inf':
                None
            else:
                #print(b[1])
                data1.append(b[0])
                data2.append(b[1])
                data1= [float(w) for w in data1]
                data2= [float(w) for w in data2]
            #print (data1)
            #data1 = (data1[1:])
        for k in arq2:
            b = k.replace(', ', ' ')
            b = b.split(' ')
            if b[0] == 'F1'or b[1]=='inf':
                None
            else:
                #print(b[1])
                data3.append(b[0])
                data4.append(b[1])
                data3 = [float(w) for w in data3]
                data4 = [float(w) for w in data4]
        for k in arq3:
            b = k.replace(', ', ' ')
            b = b.split(' ')
            if b[0] == 'F1'or b[1]=='inf':
                None
            else:
                #print(b[1])
                data5.append(b[0])
                data6.append(b[1])
                data5 = [float(w) for w in data5]
                data6 = [float(w) for w in data6]
        #print(((data5)))
        maxf1=[]
        maxn2=[]
        maxf1.append(max(data1))
        maxf1.append(max(data3))
        maxf1.append(max(data5))
        maxn2.append(max(data2))
        maxn2.append(max(data4))
        maxn2.append(max(data6))
        mn2=max(maxn2)+0.5
        mf1=max(maxf1)

        if (r == 2):
            r = 0
        if j % nb_plots_per_page == 0:
            fig = plot.figure(figsize=(8, 15), dpi=100)
        #print(j)
        # Plot stuffs !
        plot.subplot2grid(grid_size, (j % nb_plots_per_page, 0))
        plot.axis((0.06,0.2,0,3.5))
        plot.title(nome_base+" "+str(i))
        bag=plot.scatter(x=data2,y=data1,alpha=0.6,marker="v")
        dist=plot.scatter(x=data4, y=data3,alpha=0.6,marker='+')
        ag=plot.scatter(x=data6, y=data5, alpha=0.6)
        plot.xlabel("N2")
        plot.ylabel("F1")
        f=i-1
        plt.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "])
        print(mf1)
        resultado_dsoc(f)


        #Close the page if needed
        if (j + 1) % nb_plots_per_page == 0 or (j + 1) == nb_plots:
            plot.tight_layout()

            pdf_pages.savefig(fig)


        j=j+1
        r=r+1

    pdf_pages.close()

def gera_png():
    for i in range(1, 21):
        data3 = []
        data4 = []
        data1 = []
        data2 = []
        data5 = []
        data6 = []
        # arq1 = open('/media/marcos/Data/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
        arq2 = open('/media/marcos/Data/Tese/ComplexidadeDist/'+nome_base+'/' + nome_base + str(i) + '/'+nome_base+'_medias.txt', 'r')
        arq1 = open('/media/marcos/Data/Tese/ComplexidadeBags/'+nome_base+'/' + nome_base + str(i)+'/' + nome_base + '_medias.txt', 'r')
        arq3 = open('/media/marcos/Data/Tese/ComplexidadeAg/'+nome_base+'/' + nome_base + str(i)+ '/' + nome_base + '_medias.txt', 'r')

        for k in arq1:

            b = k.replace(', ', ' ')
            b=b.split(' ')
            if b[0] == 'F1':
                None
            else:
                #print(b[1])
                data1.append(b[0])
                data2.append(b[1])
                data1= [float(w) for w in data1]
                data2= [float(w) for w in data2]
            #print (data1)
            #data1 = (data1[1:])
        for k in arq2:
            b = k.replace(', ', ' ')
            b = b.split(' ')
            if b[0] == 'F1':
                None
            else:
                #print(b[1])
                data3.append(b[0])
                data4.append(b[1])
                data3 = [float(w) for w in data3]
                data4 = [float(w) for w in data4]
        for k in arq3:
            b = k.replace(', ', ' ')
            b = b.split(' ')
            if b[0] == 'F1':
                None
            else:
                #print(b[1])
                data5.append(b[0])
                data6.append(b[1])
                data5 = [float(w) for w in data5]
                data6 = [float(w) for w in data6]
        #print(((data5)))
        maxf1=[]
        maxn2=[]
        maxf1.append(max(data1))
        maxf1.append(max(data3))
        maxf1.append(max(data5))
        maxn2.append(max(data2))
        maxn2.append(max(data4))
        maxn2.append(max(data6))
        mn2=max(maxn2)
        mf1=max(maxf1)

        fig = plt.figure(figsize=(6,5), dpi=100)
        plt.axis((0.2, 1, 0, 17))
        plt.title(nome_base + " " + str(i))
        bag = plt.scatter(x=data2, y=data1, alpha=0.6, marker="v")
        dist = plt.scatter(x=data4, y=data3, alpha=0.6, marker='+')
        ag = plt.scatter(x=data6, y=data5, alpha=0.6)
        plt.xlabel("N2")
        plt.ylabel("F1")
        plt.legend([bag, dist, ag], ["BAG. ", "KNhood. ", "G.A. "])
        print(mn2)
        fig.savefig(nome_base + str(i) + ".png")
#dsoc()
gera_png()
#gera_png()
    #
    # for i in range(1,21):
    # #arq1 = open('/media/marcos/Data/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
    #     arq2 = open('/media/marcos/Data/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
    #     arq1 = open('/media/marcos/Data/Tese/ComplexidadeBags/' + nome_base + '/' + nome_base + '_medias.txt', 'r')
    #     arq3 = open('/media/marcos/Data/Tese/ComplexidadeAG/' + nome_base + '/' + nome_base + '_medias.txt', 'r')
