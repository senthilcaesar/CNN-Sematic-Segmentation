1) cp /rfanfs/pnl-zorro/projects/DIAGNOSE_CTE_U01/Data_Nipype/masking/read.sh
2) ./read.sh > qc.txt
3) sed 's/^/slicesdir -o /' qc.txt > slicesdir.sh
4) ./slicesdir.sh
