import numpy as np
import nibabel as nib
import sys

binary_file = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/binary'
training_data='/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/99percentile.npy'

modality = 'dwib0-linear-99percentile-pad.nii.gz'
def normalize_data_storage(data_storage):

    f_handle = open(binary_file, 'wb')
    count = 0
    for subject in case_arr:
        img = nib.load(subject+'/'+modality)
        imgU16 = img.get_data().astype(np.float64)
        imgU16.tofile(f_handle)
        print('Case ' + str(count) + ' done')
        count = count + 1
    f_handle.close()


cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/caselistshuff.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

normalize_data_storage(case_arr)

#x_dim=145
y_dim=256
z_dim=256
total_case = len(case_arr)
print("Merging npy files")
merge = np.memmap(binary_file, dtype=np.float64, mode='r+', shape=(142899, y_dim, z_dim))
print(merge.shape)
print("Saving training data to disk")
np.save(training_data, merge)
