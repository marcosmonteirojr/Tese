#!/usr/bin/env bash
Dir=/media/marcos/Data/Tese/Bags/16/
while read i;
do
    for j in `seq 1 100`;
    do

    if [ -e "$Dir""Individuo"$i$j".arff" ];
        then echo"";
er
        else echo  "erro$i";
    fi
    done
done<bases.txt