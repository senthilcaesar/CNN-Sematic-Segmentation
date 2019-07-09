import nibabel as nib 
import numpy as np
import cv2
orig      = nib.load('/space/dura/2/users/sq566/Downloads/ES2020_4/dwib0-linear-99percentile-pad.nii.gz').get_data()
nonbrain = np.load('/space/dura/2/users/sq566/Downloads/ES2020_4/nonBrain.npy')
recons    = np.load('/space/dura/2/users/sq566/Downloads/ES2020_4/recons.npy')
mask      = np.load('/space/dura/2/users/sq566/Downloads/ES2020_4/predict.npy')
mask[mask < 1] = 0

m,n = nonbrain.shape[::2]
nonbrain = nonbrain.transpose(0,3,1,2).reshape(m,-1,n)
recons = recons.transpose(0,3,1,2).reshape(m,-1,n)
mask = mask.transpose(0,3,1,2).reshape(m,-1,n)


print(orig.shape,mask.shape)
for i in range(0,len(orig)):
    cv2.imwrite("/homes/1/sq566/Desktop/ES2020_4/b0/"+str(i)+".png",orig[i]*255.0)
    cv2.imwrite("/homes/1/sq566/Desktop/ES2020_4/mask/"+str(i)+".png",mask[i]*255.0)
    cv2.imwrite("/homes/1/sq566/Desktop/ES2020_4/nonBrain/"+str(i)+".png",nonbrain[i]*255.0)
    cv2.imwrite("/homes/1/sq566/Desktop/ES2020_4/recons/"+str(i)+".png",recons[i]*255.0)
