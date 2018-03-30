#!/bin/bash

while read i;
do
	~/anaconda3/envs/tese2/bin/python ./GA.py $i
done < ./Bases/bases6.txt