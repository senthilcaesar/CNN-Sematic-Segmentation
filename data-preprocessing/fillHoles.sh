#!/bin/bash


filename=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/brats/caselist_qc.txt

i=1
while read -r case
do
	
	ImageMath 3 ${case}/truth-nn-pad-filled.nii.gz FillHoles ${case}/truth-nn-pad.nii.gz 
	maskfilter -force -scale 5 ${case}/truth-nn-pad-filled.nii.gz clean ${case}/truth-nn-pad-filled-cleaned.nii.gz

	echo "Case ${i} done"
	i=$((i+1))
done < ${filename}
