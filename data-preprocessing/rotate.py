import numpy as np
import nibabel as nib
import sys
import scipy.ndimage

filename='y_train_99.npy'


imgU16 = np.load(filename, mmap_mode='r')

data = scipy.ndimage.rotate(imgU16, 180, axes=(0,1))

np.save('y_train_99_rotated.npy', data)
