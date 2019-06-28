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
	
	echo "Extracting b0 ${i} .... $case"

	dim1=`fslinfo dwib0.nii.gz | grep -w dim1 | awk '{print $ NF}'`
	
	ResampleImage 3 dwib0.nii.gz dwib0-linear.nii.gz ${dim1}x246x246 1
	ResampleImage 3 truth.nii.gz truth-nn.nii.gz ${dim1}x246x246 1 1
	
	echo "Case ${i} done"
	i=$((i+1))
	cd ${base_dir}

done < ${filename}
