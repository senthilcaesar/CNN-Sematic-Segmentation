#!/bin/bash


base_dir=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original
filename=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/case.txt
echo "Total cases = `cat $filename | wc -l`"
echo

i=1
while read -r case
do
	#if [[ ! -d $case/diff ]]; then
   	#	 echo $case
	#fi
	cd ${case}
	
	unu permute -p 1 2 3 0 -i *.nhdr -o test.nhdr
	
	echo "Case ${i} done"
	i=$((i+1))
	cd ${base_dir}

done < ${filename}
