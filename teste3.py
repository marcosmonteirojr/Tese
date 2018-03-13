import os, shutil

caminho_todas = "/media/marcos/Data/Tese/AG/"
caminho_valida = "/media/marcos/Data/Tese/Bases/Validacao/"
nome_base = 'Wine'
repeticao = 1
geracao = 0
population=[[34], [67], [115], [134], [42], [133], [8], [83], [43], [71], [80], [60], [36], [86], [21], [161], [188], [19], [39], [81], [98], [33], [46], [56], [143], [183], [124], [156], [96], [158], [103], [95], [149], [82], [117], [163], [1], [108], [3], [79], [70], [53], [65], [182], [68], [52], [176], [159], [69], [41], [73], [57], [85], [40], [14], [126], [37], [29], [28], [112], [31], [75], [35], [11], [9], [84], [45], [66], [173], [100], [49], [110], [87], [94], [54], [164], [51], [132], [17], [174], [162], [145], [121], [129], [128], [88], [91], [5], [58], [62], [144], [6], [118], [198], [184], [48], [185], [7], [157], [160]]
pasta = caminho_todas + str(repeticao) + "/" + str(geracao)
pasta2 = caminho_todas + str(repeticao) + "/" + str(geracao + 1)

if (os.path.exists(pasta2) == False):
    os.system("mkdir -p " + pasta2)

for i in population:
    print(population)
    shutil.copy2(pasta+"/Individuo" + nome_base + str(i[0]) + '.arff', pasta2)
    # print(pasta+"/Individuo" + nome_base + str(i[0]) + '.arff '+pasta2+"/Individuo" + nome_base + str(i[0])+'.arff')
    #os.system("cp " + pasta + "/Individuo" + nome_base + str(i[0]) + '.arff ' + pasta2 + "/Individuo" + nome_base + str(
     #   i[0]) + '.arff')