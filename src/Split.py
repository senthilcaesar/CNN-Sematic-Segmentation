import numpy as np

X = '../data/data.npy'
Y = '../data/label.npy'

X = np.load(X)
Y = np.load(Y)

# Splitting the Dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

np.save('../data/x_train.npy', X_train)
np.save('../data/y_train.npy', Y_train)
np.save('../data/x_test.npy', X_test)
np.save('../data/y_test.npy', Y_test
