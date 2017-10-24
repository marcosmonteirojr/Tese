#!/bin/bash

for i in `seq 1 100`; do if [ -e "nomedabase"$i".arff" ]; then echo ""; else echo $i; done
