#!/bin/bash

while read i;
do
	~/anaconda3/envs/tese2/bin/python ./DcolArff.py $i
done < bases10-todasmenosmagicewdvg.txt