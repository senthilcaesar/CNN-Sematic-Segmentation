import numpy as np 
import nibabel as nib
import sys
import scipy.ndimage


img = nib.load('dwib0_X145-linear.nii.gz')
imgU16 = img.get_data().astype(np.float64)


data = scipy.ndimage.rotate(imgU16, 30, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_30.nii.gz')

data = scipy.ndimage.rotate(imgU16, 60, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_60.nii.gz')

data = scipy.ndimage.rotate(imgU16, 90, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_90.nii.gz')

data = scipy.ndimage.rotate(imgU16, 120, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_120.nii.gz')

data = scipy.ndimage.rotate(imgU16, 150, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_150.nii.gz')

data = scipy.ndimage.rotate(imgU16, 180, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_180.nii.gz')

data = scipy.ndimage.rotate(imgU16, 210, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_210.nii.gz')

data = scipy.ndimage.rotate(imgU16, 240, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_240.nii.gz')

data = scipy.ndimage.rotate(imgU16, 270, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_270.nii.gz')

data = scipy.ndimage.rotate(imgU16, 300, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_300.nii.gz')

data = scipy.ndimage.rotate(imgU16, 330, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_330.nii.gz')

data = scipy.ndimage.rotate(imgU16, 360, axes=(1,2), reshape=False)
data_nii = nib.Nifti1Image(data, img.affine, img.header)
nib.save(data_nii, 'two_360.nii.gz')
