#!/bin/bash

while read i; 
do
	~/anaconda3/envs/tese2/bin/python ./ManipulaColex.py $i
done < Bases/bases5.txt
