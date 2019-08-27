import os
import sys
import numpy as np
from keras import backend as K
import nibabel as nib
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress TensorFlow logs


cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/test/caselist.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    '''Builds and return a data flow graph
    A Tensor is a symbolic handle to one of the outputs of an operation
    It does not hold the value of that operatin out, but instead provides a means of computing the value in a TensorFlow session'''
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


for name in case_arr:
	y_true = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/test/' + name + '/truth-pad.nii.gz'
	img = nib.load(y_true)
	true_mask = img.get_data().astype(np.float32)
	y_pred = np.load('/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/test/' + name + '/SO.npy')

	high_values_flags = true_mask > 1.0
        true_mask[high_values_flags] = 1.0

	score = dice_coef(true_mask, y_pred)
	# Construct a `Session` to execute the graph.
	sess = tf.Session()
	# Execute the graph and store the value that `e` represents in `result`.
	result = sess.run(score)
	print 'Dice score: ', result, ' subject: ', name #print y_pred[77][67][50], true_mask[77][67][50]
