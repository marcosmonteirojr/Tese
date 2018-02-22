import subprocess, numpy



proc = subprocess.Popen(["find /media/marcos/Data/Tese/AG/1/1/  -iname IndividuoWine*.arff"],
                        stdout=subprocess.PIPE, shell=True)

(cont_arq, err) = proc.communicate()

cont_arq = (cont_arq.decode("utf-8"))
cont_arq = cont_arq.split()
for i in cont_arq:
    print(i)