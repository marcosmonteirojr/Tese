#!/bin/bash

#while read i;
#do
#	~/anaconda3/envs/tese2/bin/python ./GA.py $i
#done < ./Bases/bases5.txt

while read i;
do
	~/miniconda3/envs/l/bin/python ./smd3.py $i

done < ./bases27
