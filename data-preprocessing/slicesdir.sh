#!/bin/bash


filename=/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/brats/testing.txt

while read -r case
do
	
	echo -n e ${case}/dwib0-linear-99percentile-pad.nii.gz ${case}/truth-nn-pad-filled-cleaned.nii.gz

done < ${filename}
