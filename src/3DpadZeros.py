import nibabel as nib
import numpy as np

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/test/HCP/HCP.txt'

with open(cases) as f:
    case_arr = f.read().splitlines()

count = 0
for subject in case_arr:
    dwi_load = nib.load(subject+ '/out-linear.nii.gz')
    
    #print(dwi_load.get_data_dtype())
    #print(mask_load.get_data_dtype())
    
    img_dwi = dwi_load.get_data().astype(np.int16)
    img_dwi = img_dwi[0:145,:,0:86]
    
    # ((top, bottom), (left, right))
    npad = ((0, 0), (0, 0), (10, 0))
    b = np.pad(img_dwi, pad_width=npad, mode='constant', constant_values=0) 
    image = nib.Nifti1Image(b, dwi_load.affine, dwi_load.header)
    nib.save(image , subject+ '/' + 'out-linear-pad.nii.gz') 
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
