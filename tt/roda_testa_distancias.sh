#!/bin/bash

while read i; 
do
	~/anaconda3/envs/tese2/bin/python ./Vizinhos_por_lable.py $i
done < Bases/bases6.txt
