import nibabel as nib
import numpy as np
import os

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/caselistNew.txt'

with open(cases) as f:
    case_arr = f.read().splitlines()

count = 0
for subject in case_arr:
    dwi_image = nib.load(subject+ '/dwib0-linear-99percentile.nii.gz')
    dwi_mask = nib.load(subject+ '/truth-nn.nii.gz')
    
    #print(dwi_load.get_data_dtype())
    #print(mask_load.get_data_dtype())
    
    img_dwi = dwi_image.get_data().astype(np.float64)
    img_mask = dwi_mask.get_data().astype(np.float32)
    
    print("Old DWI Shape = ", img_dwi.shape)
    print("Old Mask Shape = ", img_mask.shape)

    # ((top, bottom), (left, right))
    npad = ((0, 0), (5, 5), (5, 5))
    image = np.pad(img_dwi, pad_width=npad, mode='constant', constant_values=0) 
    mask = np.pad(img_mask, pad_width=npad, mode='constant', constant_values=0) 
    
    print("New DWI Shape = ", image.shape)
    print("New Mask Shape = ", mask.shape)
    
    image_dwi = nib.Nifti1Image(image, dwi_image.affine, dwi_image.header)
    image_mask = nib.Nifti1Image(mask, dwi_mask.affine, dwi_mask.header)
    
    nib.save(image_dwi , subject+ '/' + 'dwib0-linear-99percentile-pad.nii.gz') 
    nib.save(image_mask , subject+ '/' + 'truth-nn-pad.nii.gz') 
    print("Case ", count , "done")
    count = count + 1


#---------------------------------------------------------- Padding Example -----------------------------------------------------------------------#

#A = np.array([[1,2],[3,4]])

#np.pad(A, ((1,2),(2,1)), 'constant')

#array([[0, 0, 0, 0, 0],           # 1 zero padded to the top
#       [0, 0, 1, 2, 0],           # 2 zeros padded to the bottom
#       [0, 0, 3, 4, 0],           # 2 zeros padded to the left
#       [0, 0, 0, 0, 0],           # 1 zero padded to the right
#       [0, 0, 0, 0, 0]]


#a = np.ones((4, 3, 2))

# npad is a tuple of (n_before, n_after) for each dimension
#npad = ((0, 0), (1, 2), (2, 1))
#b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)

#print(b.shape)
# (4, 6, 5)

#print(b)
# [[[ 0.  0.  0.  0.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.]]

#  [[ 0.  0.  0.  0.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.]]

#  [[ 0.  0.  0.  0.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.]]

#  [[ 0.  0.  0.  0.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  1.  1.  0.]
#   [ 0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.]]]
