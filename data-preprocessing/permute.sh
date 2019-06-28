#!/bin/bash


base_dir=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/HCP_Psychosis
filename=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/HCP_Psychosis/caselist.txt
echo "Total cases = `cat $filename | wc -l`"
echo

i=1
while read -r case
do
	cd ${case}
	
	unu permute -p 1 2 3 0 -i ${case}_3T_DWI_dir107_AP.nrrd -o ${case}_permute_AP107.nrrd
	unu permute -p 1 2 3 0 -i ${case}_3T_DWI_dir107_PA.nrrd -o ${case}_permute_PA107.nrrd
	unu permute -p 1 2 3 0 -i ${case}_3T_DWI_dir99_AP.nrrd -o ${case}_permute_AP99.nrrd
	unu permute -p 1 2 3 0 -i ${case}_3T_DWI_dir99_PA.nrrd -o ${case}_permute_PA99.nrrd
	
	echo "Case ${i} done"
	i=$((i+1))
	cd ${base_dir}

done < ${filename}
