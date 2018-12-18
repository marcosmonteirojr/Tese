from scipy.stats import  friedmanchisquare
import numpy, csv, pandas


def open_csv(base_name, arq):
    a=[]
    b=[]
    c=[]
    d=[]
    e=[]
    file = '/media/marcos/Data/Tese/Resultados Complexidade/'+base_name+"/csv_fridman/"+arq+".csv"
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a.append(row["a"])
            b.append(row['b'])
            c.append(row["c"])
            d.append(row["d"])
            e.append(row["e"])
    return a,b,c,d,e

def new_result(base_name):
    #base_name="Banana"
    x=open('/media/marcos/Data/Tese/Resultados Complexidade/'+base_name+"/Resultados_"+base_name+"_repeticao.csv","r")
    f1 =[]
    f1v = []
    f2 = []
    f3 = []
    f4 = []
    n1 =[]
    n2 = []
    n3 = []
    n4 =[]
    t1=[]
    lsc=[]
    l1=[]
    l2=[]
    l3=[]
    c1=[]
    c2=[]
    den=[]
    cls=[]
    hub=[]
    for y in x:

        y=y.split(',')
        if(y[0]!="overlapping.F1"):
            f1.append(y[0])
        if(y[1]!="overlapping.F1v"):
            f1v.append(y[1])
        if(y[2]!="overlapping.F2"):
            f2.append(y[2])
        if(y[3]!="overlapping.F3"):
            f3.append(y[3])
        if (y[4] != "overlapping.F4"):
            f4.append(y[4])
        if(y[5]!="neighborhood.N1"):
            n1.append(y[5])
        if (y[6] != "neighborhood.N2"):
            n2.append(y[6])
        if (y[7] != "neighborhood.N3"):
            n3.append(y[7])
        if (y[8] != "neighborhood.N4"):
            n4.append(y[8])
        if (y[9] != "neighborhood.T1"):
            t1.append(y[9])
        if (y[10] != "neighborhood.LSCAvg"):
            lsc.append(y[10])
        if (y[11] != "linearity.L1"):
            l1.append(y[11])
        if (y[12] != "linearity.L2"):
            l2.append(y[12])
        if (y[13] != "linearity.L3"):
            l3.append(y[13])
        if (y[17] != "balance.C1"):
            c1.append(y[17])
        if (y[18] != "balance.C2"):
            c2.append(y[18])
        if (y[19] != "network.Density"):
            den.append(y[19])
        if (y[20] != "network.ClsCoef"):
            cls.append(y[20])
        if (y[21] != "network.Hubs"):
            hub.append(y[21])
    return   f1 ,f1v, f2 , f3 , f4 , n1 , n2 ,n3 ,n4, t1,lsc, l1, l2, l3, c1, c2, den, cls,hub

def div_col(lista, base_name, arq):
    a=[]
    for i in range(0,100):    #exit(0)
            j=i
            a.append(lista[j])
            j=i
            a.append(lista[j+100])
            j = i
            a.append(lista[j + 200])
            j = i
            a.append(lista[j + 300])
            j = i
            a.append(lista[j + 400])

    b=["a","b","c","d", "e"]
    x=chunks(a,5)

    with open("/media/marcos/Data/Tese/Resultados Complexidade/"+base_name+"/csv_fridman/"+str(arq)+".csv", 'w') as g:
        w = csv.writer(g)
        w.writerow(b)
        w.writerows(x)
    del b, a, x

def chunks(lista, n):
    for i in range(0, len(lista), n):
        yield lista[i:i + n]

def exec_csv(base_name):
    f1, f1v, f2, f3, f4, n1, n2, n3, n4, t1, lsc, l1, l2, l3, c1, c2, den, cls, hub = new_result(base_name)
    div_col(f1, base_name, 1)
    div_col(f1v, base_name, 2)
    div_col(f2, base_name, 3)
    div_col(f3, base_name, 4)
    div_col(f4, base_name, 5)
    div_col(n1, base_name, 6)
    div_col(n2, base_name, 7)
    div_col(n3, base_name, 8)
    div_col(n4, base_name, 9)
    div_col(t1, base_name, 10)
    div_col(lsc, base_name, 11)
    div_col(l1, base_name, 12)
    div_col(l2, base_name, 13)
    div_col(l3, base_name, 14)
    div_col(c1, base_name, 15)
    div_col(c2, base_name, 16)
    div_col(den, base_name, 17)
    div_col(cls, base_name, 18)
    div_col(hub, base_name, 19)

def main():
    base_name="Haberman"
    exec_csv(base_name)
    p_value=[]
    #head=["Dataset","F1", "F1v", "F2", "F3",  "F4", "N1", "N2", "N3", "N4", "T1", "LSC", "L1", "L2", "L3", "C1", "C2", "Den", "CLS"
    #    , "HUB"]
    p_value.append(base_name)
    for i in range(1,20):
        a,b,c,d,e=open_csv(base_name,str(i) )
        q,p=friedmanchisquare(a,b,c,d,e)
        p_value.append(round(p,2))
        alpha = 0.05

        if p > alpha:
            print('Same distributions (fail to reject H0)')
        else:
            print('Different distributions (reject H0)')
    with open("/media/marcos/Data/Tese/Resultados Complexidade/Result.csv",
              'a') as g:
        w = csv.writer(g)
    #    w.writerow(head)
        w.writerow(p_value)

if __name__ == "__main__":
    main()