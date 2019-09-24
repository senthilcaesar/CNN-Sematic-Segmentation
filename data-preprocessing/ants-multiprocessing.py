import os
import subprocess
import nibabel as nib
import numpy as np
import time
import multiprocessing

def ANTS_rigid_body_trans(b0_nii, return_dict):
    print("Performing ants rigid body transformation...")
    input_file = b0_nii
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - 7] + '-'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    reference = '/rfanfs/pnl-zorro/home/sq566/CompNetPipeline/reference/eight256.nii.gz'

    trans_matrix = "antsRegistrationSyNQuick.sh -d 3 -f " + reference + " -m " + input_file + " -t r -o " + output_file
    output1 = subprocess.check_output(trans_matrix, shell=True)

    omat_name = case_name[:len(case_name) - 7] + '-0GenericAffine.mat'
    omat_file = os.path.join(os.path.dirname(input_file), omat_name)

    output_name = case_name[:len(case_name) - 7] + '-Warped.nii.gz'
    transformed_file = os.path.join(os.path.dirname(input_file), output_name)

    #print "output_file = ", transformed_file
    #print "omat_file = ", omat_file
    return_dict[b0_nii] = (transformed_file, omat_file)
    #return transformed_file, omat_file

cases = '/rfanfs/pnl-zorro/home/sq566/CompNetPipeline/rigid/combine/cases.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

if __name__ == '__main__':
    starttime = time.time()
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    for i in range(0,10):
        p = multiprocessing.Process(target=ANTS_rigid_body_trans, args=(case_arr[i],return_dict))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()

    print return_dict.values()

    print('That took {} seconds'.format(time.time() - starttime))
