import numpy as np
import nibabel as nib
import sys

binary_file = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/binary'
training_data='/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/train_allview_1022.npy'

modality = 'dwib0-linear-99percentile-pad.nii.gz'
def normalize_data_storage(data_storage):

    f_handle = open(binary_file, 'wb')
    count = 0
    for subject in case_arr:
        img = nib.load(subject+'/'+modality)
        
        imgU16_sagittal = img.get_data().astype(np.float32) # sagittal view

        imgU16_coronal = np.swapaxes(imgU16_sagittal,0,1) # coronal view

        imgU16_axial = np.swapaxes(imgU16_sagittal,0,2) # Axial view

        imgU16_sagittal.tofile(f_handle)
        imgU16_coronal.tofile(f_handle)
        imgU16_axial.tofile(f_handle)
        
        print('Case ' + str(count) + ' done')
        count = count + 1
    f_handle.close()


cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/brats/goodMasks_shuffled.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

normalize_data_storage(case_arr)

x_dim=661956
y_dim=256
z_dim=256
total_case = len(case_arr)
print("Merging npy files")
merge = np.memmap(binary_file, dtype=np.float32, mode='r+', shape=(x_dim, y_dim, z_dim))
print(merge.shape)
print("Saving training data to disk")
np.save(training_data, merge
