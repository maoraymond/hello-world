#!/bin/bash

for i in $(seq 1 123); do
        echo $i
        echo "#PBS -N train$i" > launch$i
        echo "cd /home/u12784/bio/maor/train" >> launch$i
        echo "/home/u12784/time python model.py TF${i}_pbm.txt `ls TF${i}_selex*.txt`" | tr "\n" " " >> launch$i
        echo "" >> launch$i
        qsub launch$i
done
