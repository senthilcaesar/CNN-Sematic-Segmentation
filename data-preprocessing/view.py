import numpy as np
import shutil
import sys

import nibabel as nib



cases = 'testcase.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

count = 0
for name in case_arr:
	#data = np.load(name + '/dwib0.npy')
	#truth = np.load(name + '/truth.npy')
	predict = np.load(name + '/predict-gaussian.npy')

	#subject_data = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/test_validation/' + name + '/dwib0-linear-99percentile-pad.nii.gz'
	subject_label = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/test_BICEPS/' + name + '/truth-nn-pad.nii.gz'


	#data_dwi = nib.load(subject_data)
	data_label = nib.load(subject_label)

	#image_data = nib.Nifti1Image(data, data_dwi.affine, data_dwi.header)
	#image_truth = nib.Nifti1Image(truth, data_label.affine, data_label.header)
	image_predict = nib.Nifti1Image(predict, data_label.affine, data_label.header)

	#nib.save(image_data, name + '/dwib0.nii')
	#nib.save(image_truth, name + '/truth.nii')
	nib.save(image_predict, name + '/predict-gaussian.nii')
	print("case ", count, " done")
	count += 1
