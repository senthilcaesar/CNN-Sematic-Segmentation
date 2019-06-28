#!/bin/bash


base_dir=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/HCP_Psychosis
filename=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/HCP_Psychosis/caselist.txt
echo "Total cases = `cat $filename | wc -l`"
echo

i=1
while read -r case
do
	cd ${case}
	
	bse.sh -i ${case}_permute_AP107.nrrd -o dwib0_AP107.nrrd
	bse.sh -i ${case}_permute_PA107.nrrd -o dwib0_PA107.nrrd
	bse.sh -i ${case}_permute_AP99.nrrd -o dwib0_AP99.nrrd
	bse.sh -i ${case}_permute_PA99.nrrd -o dwib0_PA99.nrrd
	
	echo "Case ${i} done"
	i=$((i+1))
	cd ${base_dir}

done < ${filename}
