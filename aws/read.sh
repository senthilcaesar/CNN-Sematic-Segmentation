file1='b0.txt'

file2='b0_masks.txt'

paste $file1 $file2 | while IFS="$(printf '\t')" read -r f1 f2
do
  echo -n $f1 $f2 " "
done
