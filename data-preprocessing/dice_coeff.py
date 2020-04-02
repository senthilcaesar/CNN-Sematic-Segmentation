import os
import sys
import numpy as np
from keras import backend as K
import nibabel as nib
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress TensorFlow logs


cases_truth = 'b0.txt'
with open(cases_truth) as f:
    truth_arr = f.read().splitlines()

cases_pred = 'b0_mask.txt'
with open(cases_pred) as f:
    pred_arr = f.read().splitlines()

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    '''Builds and return a data flow graph
    A Tensor is a symbolic handle to one of the outputs of an operation
    It does not hold the value of that operatin out, but instead provides a means of computing the value in a TensorFlow session'''
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


for i in range(0, len(truth_arr)):
    y_true = truth_arr[i]
    img_true = nib.load(y_true)
    true_mask = img_true.get_data().astype(np.float32)

    y_pred = pred_arr[i]
    img_pred = nib.load(y_pred)
    pred_mask = img_pred.get_data().astype(np.float32)
    
    high_values_true = true_mask > 1.0
    high_values_pred = pred_mask > 1.0
    true_mask[high_values_true] = 1.0
    pred_mask[high_values_pred] = 1.0

    score = dice_coef(true_mask, pred_mask)
    # Construct a `Session` to execute the graph.
    sess = tf.Session()
    # Execute the graph and store the value that `e` represents in `result`.
    result = sess.run(score)
    print('Dice score: ', result)
