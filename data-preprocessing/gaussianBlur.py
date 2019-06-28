import numpy as np
import nibabel as nib
import sys
import scipy.ndimage as ndimage


filename='x_train_99.npy'


imgU16 = np.load(filename, mmap_mode='r')

data = ndimage.gaussian_filter(imgU16, sigma=(0.6, 0.6, 0), order=0)

np.save('x_train_99_gaussian.npy', data)
