#!/bin/bash

while read i;
do
	~/anaconda3/envs/tese/bin/python ./ManipulaColex.py $i
done < bases2.txt
