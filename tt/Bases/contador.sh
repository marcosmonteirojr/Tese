#encontrar arquivos faltantes
!/usr/bin/env bash
#Dir=/media/marcos/Data/Tese/Bags/16/
#while read i;
#do
#    for j in `seq 1 100`;
#    do
#
#    if [ -e "$Dir""Individuo"$i$j".arff" ];
#        then echo"";
#er
#        else echo  "erro$i";
#    fi
#    done
#done<bases.txt
####################################3
###########Mudar Nomes#################3
#!/usr/bin/env bash
#Dir=/media/marcos/Data/Tese/Distancias/
#for j in `seq 1 20`;
#do
#    for k in `seq 0 171`;
#    do
#    if [ -e "$Dir"$j"/Adult/Vizinhosadult"$k".arff" ] ;
#
#        then mv "$Dir"$j"/Adult/Vizinhosadult"$k".arff" "$Dir"$j"/Adult/VizinhosAdult"$k".arff"
#        echo "alterado";
#
#fi
#done
#done
########################################################3
Dir=/media/marcos/Data/Tese/Bags/16/
#while read i;
#do
#    for j in `seq 1 20`;
#    do
#            rm "$Dir"$j"/"$i"/"*".arff.2"*
#            echo $i;
#
#
#done
#done< bases6.txt

for j in `seq 0 21`;
    do
     mv "$Dir""Adult"$j".arff" "$Dir""IndividuoAdult"$j".arff"
     echo $i;


done
