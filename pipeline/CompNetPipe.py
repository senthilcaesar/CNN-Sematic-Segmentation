# -----------------------------------------------------------------
# Author:		PNL BWH                 
# Written:		07/02/2019                             
# Last Updated: 	07/05/2019
# Purpose:  		Python pipeline for diffusion brain masking
# -----------------------------------------------------------------

import re
import os
import sys
import argparse
import os.path
import pathlib
import subprocess
import nibabel as nib
import numpy as np
from os import path
from keras.models import load_model
from keras.models import model_from_json
import cv2
import numpy as np
import sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
import numpy as np
import os
from keras import losses
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose,add,multiply
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import nibabel as nib

# parser module for input arguments
parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('-i', action='store', dest='dwi', type=str,
                    help='Input Diffusion Image')

group.add_argument('-l', type=str, dest='collection',
                    help='Input list of cases ( caselist.txt )',
                    )

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()

# suffixes
suffix_nifti    = "nii"
suffix_nifti_gz = "nii.gz"
suffix_nrrd     = "nrrd"
suffix_nhdr     = "nhdr"
suffix_npy      = "npy"

def predict_mask(input_file):
    smooth = 1.
    def dice_coef(y_true, y_pred):
        y_true_f     = K.flatten(y_true)
        y_pred_f     = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # Negative dice to obtain region of interest (ROI-Branch loss) 
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    # Positive dice to minimize overlap with region of interest (Complementary branch (CO) loss)
    def neg_dice_coef_loss(y_true, y_pred):
        return dice_coef(y_true, y_pred)

    # load json and create model
    json_file         = open('/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/model/CompNetmodel_arch_DWI_percentile_99.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model      = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/model/CompNetmodel_weights_DWI_percentile_99.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(lr=1e-5),
                  loss={'output1': dice_coef_loss, 'output2': dice_coef_loss, 'output3': dice_coef_loss,
                        'output4': dice_coef_loss, 'conv10': dice_coef_loss, 'final_op': dice_coef_loss,
                        'xoutput1': neg_dice_coef_loss, 'xoutput2': neg_dice_coef_loss, 'xoutput3': neg_dice_coef_loss,
                        'xoutput4': neg_dice_coef_loss, 'xconv10': neg_dice_coef_loss, 'xfinal_op': neg_dice_coef_loss,
                        'xxoutput1': 'mse', 'xxoutput2': 'mse', 'xxoutput3': 'mse', 'xxoutput4': 'mse',
                        'xxconv10': 'mse', 'xxfinal_op': 'mse'})

    output_file = input_file[:len(input_file) - (len(suffix_nifti_gz)+1)] + '-mask.npy'
    x_test      = nib.load(input_file).get_data()
    x_test      = x_test.reshape(x_test.shape+(1,))
    preds_train = loaded_model.predict(x_test, verbose=1)
    SO          = preds_train[5]   # Segmentation Output
    
    np.save(output_file, SO)
    return output_file

def check_gradient(Nhdr_file):
    input_file      = Nhdr_file
    bashCommand1    = "unu head " + input_file + " | grep -i sizes | awk '{print $NF}'"
    bashCommand2    = "unu head " + input_file + " | grep -i gradient | wc -l"
    output1         = subprocess.check_output(bashCommand1, shell=True)
    output2         = subprocess.check_output(bashCommand2, shell=True)
    header_gradient = int(output1.decode(sys.stdout.encoding))
    total_gradient  = int(output2.decode(sys.stdout.encoding))
    
    if header_gradient == total_gradient:
        print "Gradient check passed"
    else:
        print "Gradient check failed, Possible scenario the gradient axis is swapped"
        print 'Please check file header'
        sys.exit(1)

def resample(nii_file, dim1):
    print "Performing linear interpolation"
    input_file           = nii_file
    output_file          = input_file[:len(input_file) - (len(suffix_nifti_gz)+1)] + '-linear.nii.gz'
    bashCommand_resample = "ResampleImage 3 " + input_file + " " + output_file + " " + dim1 + "x246x246 1"
    output2              = subprocess.check_output(bashCommand_resample, shell=True)
    return output_file

def get_dimension(nii_file):
    input_file            = nii_file
    bashCommand_dimension = "fslinfo " + input_file + " | grep -w dim1 | awk '{print $ NF}'"
    process               = subprocess.Popen(bashCommand_dimension.split(), stdout=subprocess.PIPE)
    output, error         = process.communicate()
    line                  = str(output.decode(sys.stdout.encoding))
    line                  = line.split()
    index_dim1            = line.index('dim1')
    index_dim2            = line.index('dim2')
    index_dim3            = line.index('dim3')
    dim1                  = line[index_dim1+1]
    dim2                  = line[index_dim2+1]
    dim3                  = line[index_dim3+1]
    dimensions            = (dim1, dim2, dim3)
    return dimensions

def extract_b0(Nhdr_file):
    print "Extracting b0"
    input_file  = Nhdr_file
    output_file = 'dwib0_' + Nhdr_file
    bashCommand = 'bse.sh -i ' + input_file + ' -o ' + output_file + ' &>/dev/null'
    output      = subprocess.check_output(bashCommand, shell=True)
    return output_file

def nhdr_to_nifti(Nhdr_file):
    print "Converting nhdr to nifti"
    input_file    = Nhdr_file
    output_file   = input_file[:len(input_file) - len(suffix_nhdr)] + 'nii.gz'
    bashCommand   = 'ConvertBetweenFileFormats ' + input_file + " " + output_file
    process       = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output_file

def normalize(b0_resampled):
    print "Normalizing input data"
    input_file      = b0_resampled
    output_file     = input_file[:len(input_file) - (len(suffix_nifti_gz)+1)] + '-normalized.nii.gz'
    img             = nib.load(b0_resampled)
    imgU16          = img.get_data().astype(np.float64)
    p               = np.percentile(imgU16, 99)
    data            = imgU16 / p
    data[data > 1]  = 1
    data[data < 0]  = sys.float_info.epsilon
    npad            = ((0, 0), (5, 5), (5, 5))
    image           = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
    image_dwi       = nib.Nifti1Image(image, img.affine, img.header)
    nib.save(image_dwi, output_file)
    return output_file

def npy_to_nhdr(b0_normalized, dwi_mask_npy, sub_name, dim):
    image_space            = nib.load(b0_normalized)
    predict                = np.load(dwi_mask_npy)
    predict[predict < 1]   = 0
    image_predict          = nib.Nifti1Image(predict, image_space.affine, image_space.header)
    output_file            = dwi_mask_npy[:len(dwi_mask_npy) - len(suffix_npy)] + 'nii.gz'
    nib.save(image_predict, output_file)
    downsample_file        = output_file[:len(output_file) - (len(output_file)+1)] + '-downsampled.nii.gz'
    bashCommand_downsample = "ResampleImage 3 " + output_file + " " + downsample_file + " " + dim[0] + "x" + dim[1] + "x" + dim[2] + " 1"
    output2                = subprocess.check_output(bashCommand_downsample, shell=True)
    output_nhdr            = sub_name[:len(sub_name) - (len(suffix_nhdr)+1)] + '_BrainMask.nhdr'
    bashCommand            = 'ConvertBetweenFileFormats ' + downsample_file + " " + output_nhdr
    process                = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error          = process.communicate()
    return output_nhdr

# check if file exists
if args.dwi:
    f = pathlib.Path(args.dwi)
    if f.exists():
        print("File exist")
        filename = os.path.basename(args.dwi)
    else:
        print("File not found")
        sys.exit(1)


    # Input in nifti format
    if filename.endswith(suffix_nifti_gz) | filename.endswith(suffix_nifti):
        print("Nifti file, Converting nifti to nhdr format")
        input_file = filename
        if filename.endswith(suffix_nifti_gz):
            output_file = input_file[:len(input_file) - len(suffix_nifti_gz)] + 'nhdr'
            bashCommand = 'ConvertBetweenFileFormats ' + input_file + " " + output_file
        else:
            output_file = input_file[:len(input_file) - len(suffix_nifti)] + 'nhdr'
            bashCommand = 'ConvertBetweenFileFormats ' + input_file + " " + output_file

        process       = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        b0_nhdr       = extract_b0(output_file)

    # Input in nrrd / nhdr format
    elif filename.endswith(suffix_nhdr) | filename.endswith(suffix_nrrd):
        print "Nrrd / Nhdr file format"
        input_file     = filename
        check_gradient(input_file)
        b0_nhdr        = extract_b0(input_file)
        b0_nii         = nhdr_to_nifti(b0_nhdr)
        dimensions     = get_dimension(b0_nii)
        b0_resampled   = resample(b0_nii, dimensions[0])
        b0_normalized  = normalize(b0_resampled)
        dwi_mask_npy   = predict_mask(b0_normalized)
        dwi_mask       = npy_to_nhdr(b0_normalized, dwi_mask_npy, input_file, dimensions)

        for filename in os.listdir(os.getcwd()):
            if filename.startswith('dwi'):
                os.unlink(filename)

        print "Mask file = ", dwi_mask
    else:
        print "Invalid file format, Input file should be in the format *.nii.gz, *.nii, *.nrrd, *.nhdr"








# check if list of files exits
if args.collection:
    with open(str(args.collection)) as f:
        case_arr = f.read().splitlines()

    for subject in case_arr:
        img = subject + '/' + 'dwib0-linear-99percentile-pad.nii.gz'
        f = pathlib.Path(img)
        if f.exists():
            print("File exist")
        else:
            print(img, "File not found")
            sys.exit(1)
