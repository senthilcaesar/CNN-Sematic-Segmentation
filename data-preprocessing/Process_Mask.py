import numpy as np
import nibabel as nib

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/brats/goodMasks_shuffled.txt'
binary_file = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/binary_mask'
training_data='/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/trainMask_allview_1022.npy'

with open(cases) as f:
    case_arr = f.read().splitlines()

x_dim=661956
y_dim=256
z_dim=256
total_case = len(case_arr)
count=0

modality = 'truth-nn-pad-filled-cleaned.nii.gz'
f_handle = open(binary_file, 'wb')
for subjects in case_arr:
    img = nib.load(subjects+'/'+modality)
    imgU16_sagittal = img.get_data().astype(np.uint8) # sagittal view
    imgU16_sagittal[imgU16_sagittal > 0.0] = 1
    imgU16_coronal = np.swapaxes(imgU16_sagittal,0,1) # coronal view
    imgU16_axial = np.swapaxes(imgU16_sagittal,0,2) # Axial view
    imgU16_sagittal.tofile(f_handle)
    imgU16_coronal.tofile(f_handle)
    imgU16_axial.tofile(f_handle)
    print('Case ' + str(count) + ' done')
    count = count + 1
f_handle.close()

print('Merging files')
merge = np.memmap(binary_file, dtype=np.uint8, mode='r+', shape=(x_dim, y_dim, z_dim))
print(merge.shape)
print('Saving Training data to disk')
np.save(training_data, merge)
