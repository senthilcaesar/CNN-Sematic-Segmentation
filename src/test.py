import numpy as np
X = 'x_train_f32.npy'
Y = 'y_train_f32.npy'
X = np.load(X)
Y = np.load(Y)

# Splitting the Dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)

print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

np.save('x_test_f32.npy', X_test)
np.save('y_test_f32.npy', Y_test)

#-----------------------------------------------------
import numpy as np
data = np.load('y_train.npy', mmap_mode="r")
first_slice = data[0:176,:,:]
print first_slice.shape
np.save('y_train_1sub.npy', first_slice)

#------------------------------------------------------
import numpy as np
data = np.load('RO.npy')
print data.shape
import nibabel as nib
new_image = nib.Nifti1Image(data, affine=np.eye(4))
nib.save(new_image, 'RO.nii')


