#!/bin/bash


base_dir=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/HCP_Psychosis
filename=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/HCP_Psychosis/caselist.txt
echo "Total cases = `cat $filename | wc -l`"
echo

i=1
while read -r case
do
	cd ${case}
	
	ConvertBetweenFileFormats dwib0_AP107.nrrd dwib0_AP107.nii.gz
	ConvertBetweenFileFormats dwib0_PA107.nrrd dwib0_PA107.nii.gz
	ConvertBetweenFileFormats dwib0_AP99.nrrd dwib0_AP99.nii.gz
	ConvertBetweenFileFormats dwib0_PA99.nrrd dwib0_PA99.nii.gz
	
	echo "Case ${i} done"
	i=$((i+1))
	cd ${base_dir}

done < ${filename}
