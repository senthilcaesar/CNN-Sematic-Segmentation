import numpy as np 
import nibabel as nib 
import cv2
orig = nib.load('/orig.mgz').get_data()/255.0
mask = nib.load('/brainmask.mgz').get_data()
mask[mask>0]=1
print(orig.shape,mask.shape)
for i in range(0,len(orig)):
cv2.imwrite("/input/"+str(i)+".png",orig[i]*255.0)
cv2.imwrite("/mask/"+str(i)+".png",mask[i]*255.0)
