import numpy as np 
import cv2


sagittal = np.load('four-sagittal_SO.npy')
coronal = np.load('four-coronal_SO.npy')
axial = np.load('four-axial_SO.npy')

#m,n = sagittal.shape[::2]
#sagittal = sagittal.transpose(0,3,1,2).reshape(m,-1,n)
#coronal = coronal.transpose(0,3,1,2).reshape(m,-1,n)
#axial = axial.transpose(0,3,1,2).reshape(m,-1,n)


#sagittal[sagittal>0]=1
#coronal[coronal>0]=1
#coronal[coronal>0]=1

for i in range(0,len(sagittal)):
    cv2.imwrite("/Users/sq566/Desktop/sagittal/"+str(i)+".png",sagittal[i]*255.0)
    cv2.imwrite("/Users/sq566/Desktop/coronal/"+str(i)+".png",coronal[i]*255.0)
    cv2.imwrite("/Users/sq566/Desktop/axial/"+str(i)+".png",axial[i]*255.0)
