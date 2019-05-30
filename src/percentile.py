import numpy as np
import nibabel as nib
import sys

#binary_file = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/binary'
#training_data='/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/new_test.npy'

modality = 'dwib0-linear.nii.gz'
def normalize_data_storage(data_storage):

 #   f_handle = open(binary_file, 'wb')
    count = 0
    for subject in case_arr:
        img = nib.load(subject+'/'+modality)
        imgU16 = img.get_data().astype(np.float64)
        p = np.percentile(imgU16, 99)
        #print('99th Percentile = ', p, subject)
        #print('Max val = ', imgU16.max())
        data = imgU16 / p
        #print(data.max(), data.min())
        data[data > 1] = 1
        data[data < 0] = sys.float_info.epsilon
        print(data.max(), data.min())
        data_nii = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(data_nii, subject + '/' + 'dwib0-linear-99percentile.nii.gz')
       	print("case ", count, "done")
        count += 1
	
cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/caselistshuff.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

normalize_data_storage(case_arr)

#x_dim=144
#y_dim=144
#z_dim=96
#total_case = len(case_arr)
#print("Merging npy files")
#merge = np.memmap(binary_file, dtype=np.float64, mode='r+', shape=(x_dim*total_case, y_dim, z_dim))
#print(merge.shape)
#print("Saving training data to disk")
#np.save(training_data, merge)
