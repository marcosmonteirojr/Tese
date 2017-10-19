#!/bin/bash

while read i; 
do
	~/anaconda3/envs/tese1/bin/python ./ $i
done < bases.txt
