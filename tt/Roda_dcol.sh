#!/bin/bash

while read i;
do
	~/anaconda3/envs/tese/bin/python ./DcolArff.py $i
done < bases2.txt
