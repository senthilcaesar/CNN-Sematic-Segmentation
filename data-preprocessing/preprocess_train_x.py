import os
import numpy as np
import nibabel as nib

sagittal_bin_file = 'sagittal-binary-dwib0'
coronal_bin_file = 'coronal-binary-dwib0'
axial_bin_file = 'axial-binary-dwib0'

sagittal_trainingdata='sagittal-traindata-dwib0.npy'
coronal_trainingdata='coronal-traindata-dwib0.npy'
axial_trainingdata='axial-traindata-dwib0.npy'

sagittal_f_handle = open(sagittal_bin_file, 'wb')
coronal_f_handle = open(coronal_bin_file, 'wb')
axial_f_handle = open(axial_bin_file, 'wb')

def process_trainingdata(dwib0_arr):
    count = 0
    for b0 in dwib0_arr:
        img = nib.load(b0)
        imgF32 = img.get_data().astype(np.float32)
        p = np.percentile(imgF32, 99)
        imgF32_sagittal = imgF32 / p 			          # sagittal view
        imgF32_sagittal[ imgF32_sagittal < 0 ] = 0 	      
        imgF32_sagittal[ imgF32_sagittal > 1 ] = 1 		  
        imgF32_coronal = np.swapaxes(imgF32_sagittal,0,1) # coronal view
        imgF32_axial = np.swapaxes(imgF32_sagittal,0,2)   # Axial view

        imgF32_sagittal.tofile(sagittal_f_handle)
        imgF32_coronal.tofile(coronal_f_handle)
        imgF32_axial.tofile(axial_f_handle)

        print('Case ' + str(count) + ' done')
        count = count + 1

    sagittal_f_handle.close()
    axial_f_handle.close()
    coronal_f_handle.close()


cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/training_x_shuf'
with open(cases) as f:
    dwib0_arr = f.read().splitlines()

process_trainingdata(dwib0_arr)

x_dim=len(dwib0_arr)*256
y_dim=256
z_dim=256

merge_sagittal = np.memmap(sagittal_bin_file, dtype=np.float32, mode='r+', shape=(x_dim, y_dim, z_dim))
print(merge_sagittal.shape)
print("Saving sagittal training data dwib0 to disk")
np.save(sagittal_trainingdata, merge_sagittal)
os.unlink(sagittal_bin_file)

merge_coronal = np.memmap(coronal_bin_file, dtype=np.float32, mode='r+', shape=(x_dim, y_dim, z_dim))
print(merge_coronal.shape)
print("Saving coronal training data dwib0 to disk")
np.save(coronal_trainingdata, merge_coronal)
os.unlink(coronal_bin_file)

merge_axial = np.memmap(axial_bin_file, dtype=np.float32, mode='r+', shape=(x_dim, y_dim, z_dim))
print(merge_axial.shape)
print("Saving axial training data dwib0 to disk")
np.save(axial_trainingdata, merge_axial)
os.unlink(axial_bin_file)
