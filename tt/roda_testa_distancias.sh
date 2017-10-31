#!/bin/bash

while read i; 
do
	~/anaconda3/envs/tese2/bin/python ./GeraGrafico.py $i
done < bases.txt
