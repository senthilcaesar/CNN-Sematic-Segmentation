import numpy as np
import nibabel as nib

def normalize_data(data, mean, std):
    print("Mean = ", mean)
    print("SD = ", std)
    data -= mean
    data /= std
    return data

binary_file = '/rfanfs/pnl-zorro/home/sq566/OASIS/X_train/output/binary'
training_data='/rfanfs/pnl-zorro/home/sq566/OASIS/X_train/output/data.npy'

def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for subject in case_arr:
        img = nib.load(subject)
        imgU16 = img.get_data().astype(np.int16)
        means.append(imgU16.mean(axis=(0, 1, 2)))
        stds.append(imgU16.std(axis=(0, 1, 2)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    f_handle = open(binary_file, 'wb')
    count = 0
    for subject in case_arr:
        print("Normalizing ", subject)
        img = nib.load(subject)
        data = img.get_data().astype(np.float64)
        data_n = normalize_data(data, mean, std)
        data_n[data_n < 0.0] = 0;
        data.tofile(f_handle)
        print('Case ' + str(count) + ' done')
        count = count + 1
    f_handle.close()


cases = '/rfanfs/pnl-zorro/home/sq566/OASIS/X_train/case.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

normalize_data_storage(case_arr)

x_dim=176
y_dim=208
z_dim=176
total_case = len(case_arr)
merge = np.memmap(binary_file, dtype=np.float64, mode='r+', shape=(x_dim*total_case, y_dim, z_dim))
print(merge.shape)

np.save(training_data, merge)


#import pickle
#
#with open('mean', 'wb') as fp:
#    pickle.dump(means, fp)
#    
#with open('std', 'wb') as fp:
#    pickle.dump(stds, fp)
