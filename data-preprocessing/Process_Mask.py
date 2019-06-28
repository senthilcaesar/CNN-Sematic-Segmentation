import numpy as np
import nibabel as nib

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/caselistshuff.txt'
binary_file = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/binary_mask'
training_data='/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/99percentileMask.npy'

with open(cases) as f:
    case_arr = f.read().splitlines()

#x_dim=128
y_dim=256
z_dim=256
total_case = len(case_arr)
count=0

modality = 'truth-nn-pad.nii.gz'
f_handle = open(binary_file, 'wb')
for subjects in case_arr:
	img = nib.load(subjects+'/'+modality)
	data = img.get_data().astype(np.float32)
	data[data>0.0]=1
	data.tofile(f_handle)
	print('Case ' + str(count) + ' done')
	count = count + 1
f_handle.close()

print('Merging files')
merge = np.memmap(binary_file, dtype=np.float32, mode='r+', shape=(142899, y_dim, z_dim))
print(merge.shape)
print('Saving Training data to disk')
np.save(training_data, merge)
