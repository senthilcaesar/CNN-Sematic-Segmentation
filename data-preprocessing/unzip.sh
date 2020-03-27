base_dir=/rfanfs/pnl-zorro/projects/Harmonization/abcd/site21/Reference
filename=/rfanfs/pnl-zorro/projects/Harmonization/abcd/site21/Reference/cases.txt
echo "Total cases = `cat $filename | wc -l`"
echo

i=1
while read -r case
do
	
	echo "Motion and Eddy Current Correction case ${i} .... $T1"
  
	tar -xvzf ${case} > /dev/null

	echo "Case ${i} done"
	i=$((i+1))

done < ${filename}
