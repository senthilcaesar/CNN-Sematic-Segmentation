# Compute Transformation matrix
flirt -in ${volA} -ref ${volB} -omat A2B.mat -dof 6 -cost mutualinfo -searchcost mutualinfo

# Apply the tranformation to Input Volume
flirt -in ${volA} -ref ${volB} -applyxfm -init A2B.mat -o AinB.nii.gz

# Invert the Matrix
convert_xfm -omat B2A.mat -inverse A2B.mat

# Apply the tranformation to the mask
flirt -in ${volB} -ref ${volA} -applyxfm -init B2A.mat -o BinA
