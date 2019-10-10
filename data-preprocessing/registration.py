import os
from multiprocessing import Process, Manager, Value, Pool
import multiprocessing as mp
import subprocess

SUFFIX_NIFTI_GZ = "nii.gz"

def FSL_rigid_body_trans(b0_nii):

    print("Performing fsl rigid body transformation...")
    input_file = b0_nii
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-transformed.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    reference = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/reference/IITmean_b0_256.nii.gz'

    #Compute Transformation matrix using flirt
    omat_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '.mat'
    omat_file = os.path.join(os.path.dirname(input_file), omat_name)

    trans_matrix = "flirt -in " + input_file +  " -ref " + reference + \
                   " -omat " + omat_file + " -dof 6 -cost mutualinfo"

    #print trans_matrix
    output1 = subprocess.check_output(trans_matrix, shell=True)

    # Apply this transformation to the input volume
    apply_trans = "flirt -in " + input_file + " -ref " + reference + \
                  " -applyxfm -init " + omat_file + " -o " + output_file

    #print apply_trans
    output2 = subprocess.check_output(apply_trans, shell=True)
    return (output_file, omat_file)

def ANTS_rigid_body_trans(b0_nii, reference=None):

    print("Performing ants rigid body transformation...")
    input_file = b0_nii
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    if reference is None:
        reference = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/reference/IITmean_b0_256.nii.gz'

    trans_matrix = "antsRegistrationSyNQuick.sh -d 3 -f " + reference + " -m " + input_file + " -t r -o " + output_file
    output1 = subprocess.check_output(trans_matrix, shell=True)

    omat_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-0GenericAffine.mat'
    omat_file = os.path.join(os.path.dirname(input_file), omat_name)

    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-Warped.nii.gz'
    transformed_file = os.path.join(os.path.dirname(input_file), output_name)

    #print "output_file = ", transformed_file
    #print "omat_file = ", omat_file
    return (transformed_file, omat_file)

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/ANTS_TEST/test.txt'
with open(cases) as f:
	reference_list = f.read().splitlines()

p = Pool(processes=mp.cpu_count())
data = p.map(FSL_rigid_body_trans, [reference_list[i] for i in range(0, len(reference_list))])
p.close()
