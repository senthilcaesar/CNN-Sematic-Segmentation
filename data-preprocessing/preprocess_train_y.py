import os
import numpy as np
import nibabel as nib

sagittal_bin_file = 'sagittal-binary-mask'
coronal_bin_file = 'coronal-binary-mask'
axial_bin_file = 'axial-binary-mask'

sagittal_trainingdata='sagittal-traindata-mask.npy'
coronal_trainingdata='coronal-traindata-mask.npy'
axial_trainingdata='axial-traindata-mask.npy'

sagittal_f_handle = open(sagittal_bin_file, 'wb')
coronal_f_handle = open(coronal_bin_file, 'wb')
axial_f_handle = open(axial_bin_file, 'wb')

def process_trainingdata(mask_arr):
    count = 0
    for b0_mask in mask_arr:
        img = nib.load(b0_mask)
        imgU8_sagittal = img.get_data().astype(np.uint8) # sagittal view
        imgU8_sagittal[ imgU8_sagittal < 0 ] = 0        
        imgU8_sagittal[ imgU8_sagittal > 1 ] = 1        
        imgU8_coronal = np.swapaxes(imgU8_sagittal,0,1) # coronal view
        imgU8_axial = np.swapaxes(imgU8_sagittal,0,2)   # Axial view

        imgU8_sagittal.tofile(sagittal_f_handle)
        imgU8_coronal.tofile(coronal_f_handle)
        imgU8_axial.tofile(axial_f_handle)

        print('Case ' + str(count) + ' done')
        count = count + 1

    sagittal_f_handle.close()
    axial_f_handle.close()
    coronal_f_handle.close()


cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/training_y_shuf'
with open(cases) as f:
    mask_arr = f.read().splitlines()

process_trainingdata(mask_arr)

x_dim=len(mask_arr)*256
y_dim=256
z_dim=256

merge_sagittal = np.memmap(sagittal_bin_file, dtype=np.uint8, mode='r+', shape=(x_dim, y_dim, z_dim))
print(merge_sagittal.shape)
print("Saving sagittal training data mask to disk")
np.save(sagittal_trainingdata, merge_sagittal)
os.unlink(sagittal_bin_file)

merge_coronal = np.memmap(coronal_bin_file, dtype=np.uint8, mode='r+', shape=(x_dim, y_dim, z_dim))
print(merge_coronal.shape)
print("Saving coronal training data mask to disk")
np.save(coronal_trainingdata, merge_coronal)
os.unlink(coronal_bin_file)

merge_axial = np.memmap(axial_bin_file, dtype=np.uint8, mode='r+', shape=(x_dim, y_dim, z_dim))
print(merge_axial.shape)
print("Saving axial training data mask to disk")
np.save(axial_trainingdata, merge_axial)
os.unlink(axial_bin_file)
