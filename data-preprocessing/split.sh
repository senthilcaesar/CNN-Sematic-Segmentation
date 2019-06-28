import numpy as np

X = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/test_HCP_Psychosis.npy'
#Y = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/output/99percentileMask.npy'

X = np.load(X, mmap_mode="r")
#Y = np.load(Y, mmap_mode="r")


# Splitting the Dataset into Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

X_train = X[0:23520,:,:]
X_test  = X[23520:,:,:]

#Y_train = Y[0:108544,:,:]
#Y_test  = Y[108544:,:,:]

print(X_train.shape)
print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

print('Saving the training data.....')
np.save('../data/output/test_HCP_one.npy', X_train)
#np.save('../data/train/y_train_99.npy', Y_train)
np.save('../data/output/test_HCP_two.npy', X_test)
#np.save('../data/train/y_test_99.npy', Y_test)
