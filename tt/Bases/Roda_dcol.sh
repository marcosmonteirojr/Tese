#!/bin/bash

while read i;
do
	~/anaconda3/envs/tese2/bin/pythonsmd3.py $i
done < bases2.txt