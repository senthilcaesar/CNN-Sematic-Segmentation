import numpy as np


train_data = 'x_train_G.npy'
train_label = 'y_train_G.npy'

tr_d = np.load(train_data, mmap_mode='r')
tr_l = np.load(train_label, mmap_mode='r')

randomize = np.arange(int(tr_d.shape[0]))


np.random.shuffle(randomize)

tr_d = tr_d[randomize,:,:]
tr_l = tr_l[randomize,:,:]

np.save('x_train_99_G_shuffled.npy', tr_d)
np.save('y_train_99_G_shuffled.npy', tr_l)
