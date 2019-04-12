import numpy as np
import nibabel as nib

cases = 'case.txt'
binary_file = 'output/binary'
training_data='output/data_5089.npy'

with open(cases) as f:
    case_arr = f.read().splitlines()

x_dim=176
y_dim=208
z_dim=176
total_case = len(case_arr)
count=0

f_handle = open(binary_file, 'wb')
for subjects in case_arr:
     img = nib.load(subjects)
     imgU16 = img.get_data().astype(np.float64)    # Signed 16-bit Integer (-32768 to 32767)
     imgU16 = imgU16.reshape(x_dim, y_dim, z_dim)
     '''Considering the maximum value among all your images and the minimun one. 
     This equation will put all your images within the same range of variation. 
     Apply it to all the images (x-min(x))/(max(x)-min(x))'''
     normalized = imgU16/5089.0
     normalized.tofile(f_handle)
     print('Case ' + str(count) + ' done')
     count = count + 1
f_handle.close()

merge = np.memmap(binary_file, dtype=np.float64, mode='r', shape=(x_dim*total_case, y_dim, z_dim))
print("Saving the file")
np.save(training_data, merge)
	
