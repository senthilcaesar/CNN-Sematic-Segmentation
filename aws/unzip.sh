filename=/home/ec2-user/test_abcd/tar_cases.txt
echo "Total cases = `cat $filename | wc -l`"
echo

i=1
while read -r case
do
	tar -xvzf ${case} > /dev/null
	rm ${case}

done < ${filename}
