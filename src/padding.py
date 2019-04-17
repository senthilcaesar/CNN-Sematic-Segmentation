import nibabel as nib
import numpy as np

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/caselistfinal.txt'

with open(cases) as f:
    case_arr = f.read().splitlines()

count = 0
for subject in case_arr:
    dwi_load = nib.load(subject+ '/dwib0.nii.gz')
    mask_load = nib.load(subject+ '/truth.nii.gz')
    
    #print(dwi_load.get_data_dtype())
    #print(mask_load.get_data_dtype())
    
    img_dwi = dwi_load.get_data().astype(np.int16)
    img_mask = mask_load.get_data().astype(np.float32)
    
    #result_dwi = np.zeros((144,144,96))
    result_mask = np.zeros((144,144,96))
    
    result_dwi[:img_dwi.shape[0],:img_dwi.shape[1],:img_dwi.shape[2]] = img_dwi
    result_mask[:img_mask.shape[0],:img_mask.shape[1],:img_mask.shape[2]] = img_mask
    
    
    image = nib.Nifti1Image(result_dwi, dwi_load.affine, dwi_load.header)
    label = nib.Nifti1Image(result_mask, mask_load.affine, mask_load.header)
    
    nib.save(image , subject+ '/' + 'dwib0-pad.nii.gz')
    nib.save(label , subject+ '/' + 'truth-pad.nii.gz')
    
    print("Case ", count , "done")
    count = count + 1
