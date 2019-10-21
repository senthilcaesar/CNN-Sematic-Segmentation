import os
from multiprocessing import Process, Manager, Value, Pool
import multiprocessing as mp
import subprocess

SUFFIX_NIFTI_GZ = "nii.gz"


def ANTS_rigid_body_trans(b0_nii, mask_file):

    print("Performing ants rigid body transformation...")
    input_file = b0_nii
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    reference = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/reference/IITmean_b0_256.nii.gz'

    trans_matrix = "antsRegistrationSyNQuick.sh -d 3 -f " + reference + " -m " + input_file + " -t r -o " + output_file
    output1 = subprocess.check_output(trans_matrix, shell=True)

    omat_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-0GenericAffine.mat'
    omat_file = os.path.join(os.path.dirname(input_file), omat_name)

    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-Warped.nii.gz'
    transformed_file = os.path.join(os.path.dirname(input_file), output_name)

    #print "output_file = ", transformed_file
    #print "omat_file = ", omat_file
    return (transformed_file, omat_file, mask_file)

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/INTRUST/dwi.txt'
with open(cases) as f:
	dwi_list = f.read().splitlines()

masks = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/INTRUST/masks.txt'
with open(masks) as f:
	mask_list = f.read().splitlines()

dwi_masks = []

for i in range(0, len(dwi_list)):
    tup = (dwi_list[i], mask_list[i])
    dwi_masks.append(tup)

p = Pool(processes=mp.cpu_count())
data = p.starmap(ANTS_rigid_body_trans, dwi_masks)
p.close()

transformed_cases = []
omat_list = []
masks_new_list = []

for subject_ANTS in data:
    transformed_cases.append(subject_ANTS[0])
    omat_list.append(subject_ANTS[1])
    masks_new_list.append(subject_ANTS[2])

# Apply the same tranformation to the mask file
for i in range(0, len(transformed_cases)):

    input_file = transformed_cases[i]
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-mask.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)
    apply_mask_trans = "antsApplyTransforms -d 3 -i " + masks_new_list[i] + " -r " + input_file + " -o " \
                            + output_file + " --transform [" + omat_list[i] + "]"

    output2 = subprocess.check_output(apply_mask_trans, shell=True)
