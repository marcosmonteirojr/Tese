#!/bin/bash

while read i; 
do
	~/anaconda3/envs/tese2/bin/python ./GraficoComplex.py $i
done < Bases/bases2.txt
