import numpy as np

# Stack 3d array

x_train_1 = np.load('x_train_99.npy', mmap_mode='r')
x_train_2 = np.load('x_train_99_gaussian.npy', mmap_mode='r')
#x_train_3 = np.load('x_train_99_rotated.npy', mmap_mode='r')


y_train_1 = np.load('y_train_99.npy', mmap_mode='r')
y_train_2 = np.load('y_train_99_gaussian.npy', mmap_mode='r')
#y_train_3 = np.load('y_train_99_rotated.npy', mmap_mode='r')


x_train_GR = np.concatenate((x_train_1, x_train_2[0:36181,:,:]), axis=0)
y_train_GR = np.concatenate((y_train_1, y_train_2[0:36181,:,:]), axis=0)

#np.concatenate((x1, x2), axis=0)

print(x_train_GR.shape)
print(y_train_GR.shape)

np.save('x_train_G.npy', x_train_GR)
np.save('y_train_G.npy', y_train_GR)
