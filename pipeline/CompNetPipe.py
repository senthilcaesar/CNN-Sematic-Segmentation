# -----------------------------------------------------------------
# Author:		PNL BWH                 
# Written:		07/02/2019                             
# Last Updated: 	07/08/2019
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow message
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

parser.add_argument('-i', action='store', dest='dwi', type=str,
                    help='Input Diffusion Image')

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
suffix_txt      = "txt"

def predict_mask(input_file):
    '''
    Parameters
    ----------
    input_file : str
                 (single case filename which is stored in disk in *.nii.gz format) or 
                 (list of cases, all appended to 3d numpy array stored in disk in *.npy format)
    
    Returns
    -------
    output_file : str
                  returns the neural network predicted filename which is stored
                  in disk in 3d numpy array *.npy format
    '''
    print "Loading tensorflow ..."
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

    if input_file.endswith(suffix_nifti_gz):
        x_test      = nib.load(input_file).get_data()
    else:
        x_test      = np.load(input_file)

    x_test          = x_test.reshape(x_test.shape+(1,))
    preds_train     = loaded_model.predict(x_test, verbose=1)
    SO              = preds_train[5]   # Segmentation Output
    
    np.save(output_file, SO)
    return output_file

def check_gradient(Nhdr_file):
    '''
    Parameters
    ----------
    Nhdr_file : str
                Accepts Input filename in Nhdr format

    Returns
    -------
    None
    '''
    input_file      = Nhdr_file
    header_gradient = 0
    total_gradient  = 1
    bashCommand1    = "unu head " + input_file + " | grep -i sizes | awk '{print $5}'"
    bashCommand2    = "unu head " + input_file + " | grep -i gradient | wc -l"
    output1         = subprocess.check_output(bashCommand1, shell=True)
    output2         = subprocess.check_output(bashCommand2, shell=True)
    if output1.strip():
        header_gradient = int(output1.decode(sys.stdout.encoding))
        total_gradient  = int(output2.decode(sys.stdout.encoding))
    
        if header_gradient == total_gradient:
            print "Gradient check passed, " , input_file
        else:
            print "Gradient check failed, " , input_file, 'Please check file header'
            sys.exit(1)
    else:
        print "Gradient check passed, " , input_file

def resample(nii_file, dim1):
    '''
    Parameters
    ----------
    nii_file    : str
                  Accepts nifti filename in *.nii.gz format
    dim1        : str
                  Dimension of "x" axis in the standard coordinate space

    Returns
    -------
    output_file : str
                  linear interpolated filename which is stored in disk in *.nii.gz format
    '''
    print "Performing linear interpolation"
    input_file           = nii_file
    output_file          = input_file[:len(input_file) - (len(suffix_nifti_gz)+1)] + '-linear.nii.gz'
    bashCommand_resample = "ResampleImage 3 " + input_file + " " + output_file + " " + dim1 + "x246x246 1"
    output2              = subprocess.check_output(bashCommand_resample, shell=True)
    return output_file

def get_dimension(nii_file):
    '''
    Parameters
    ---------
    nii_file   : str
                 Accepts nifti filename in *.nii.gz format

    Returns
    -------
    dimensions : tuple
                 Dimension of the nifti file
                 example (128,176,256)
    '''
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
    '''
    Parameters
    ---------
    Nhdr_file   : str
                  Accepts nhdr filename in *.nhdr format

    Returns
    --------
    output_file : str
                  Extracted b0 nhdr filename which is stored in disk
                  Uses "bse.sh" program
    '''
    print "Extracting b0"
    input_file  = Nhdr_file
    output_file = 'dwib0_' + Nhdr_file
    bashCommand = 'bse.sh -i ' + input_file + ' -o ' + output_file + ' &>/dev/null'
    output      = subprocess.check_output(bashCommand, shell=True)
    return output_file

def nhdr_to_nifti(Nhdr_file):
    '''
    Parameters
    ---------
    Nhdr_file   : str
                  Accepts nhdr filename in *.nhdr format

    Returns
    --------
    output_file : str
                  Converted nifti file which is stored in disk
                  Uses "ConvertBetweenFilename" program
    '''
    print "Converting nhdr to nifti"
    input_file    = Nhdr_file
    output_file   = input_file[:len(input_file) - len(suffix_nhdr)] + 'nii.gz'
    bashCommand   = 'ConvertBetweenFileFormats ' + input_file + " " + output_file
    process       = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output_file

def normalize(b0_resampled):
    '''
    Parameters
    ---------
    b0_resampled : str
                   Accepts b0 resampled filename in *.nii.gz format

    Returns
    --------
    output_file : str
                  Normalized by 99th percentile filename which is stored in disk
    '''
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

def npy_to_nhdr(b0_normalized_cases, cases_mask_arr, sub_name, dim):

    '''
    Parameters
    ---------
    b0_normalized_cases : str or list
                          str  (b0 normalized single filename which is in *.nii.gz format)
                          list (b0 normalized list of filenames which is in *.nii.gz format)
    case_mask_arr       : str or list
                          str  (single predicted mask filename which is in 3d numpy *.npy format)
                          list (list of predicted mask filenames which is in 3d numpy *.npy format)
    sub_name            : str or list
                          str  (single input case filename which is in *.nhdr format)
                          list (list of input case filename which is in *.nhdr format)
    dim                 : tuple or list of tuple
                          tuple (dimension of single case in tuple format, (128,176,256))
                          list of tuples (dimension of all cases)

    Returns
    --------
    output_mask         : str or list
                          str  (single brain mask filename which is stored in disk in *.nhdr format)
                          list (list of brain mask for all cases which is stored in disk in *.nhdr format)
    '''
    if len(b0_normalized_cases) > 1:
        output_mask = []
        for i in range(0, len(b0_normalized_cases)):
            image_space            = nib.load(b0_normalized_cases[i])
            predict                = np.load(cases_mask_arr[i])
            predict[predict < 1]   = 0
            image_predict          = nib.Nifti1Image(predict, image_space.affine, image_space.header)
            output_file            = cases_mask_arr[i][:len(cases_mask_arr[i]) - len(suffix_npy)] + 'nii.gz'
            nib.save(image_predict, output_file)
            downsample_file        = output_file[:len(output_file) - (len(output_file)+1)] + '-downsampled.nii.gz'
            bashCommand_downsample = "ResampleImage 3 " + output_file + " " + downsample_file + " " + dim[i][0] + "x" + dim[i][1] + "x" + dim[i][2] + " 1"
            output2                = subprocess.check_output(bashCommand_downsample, shell=True)
            output_nhdr            = sub_name[i][:len(sub_name[i]) - (len(suffix_nhdr)+1)] + '_BrainMask.nhdr'
            bashCommand            = 'ConvertBetweenFileFormats ' + downsample_file + " " + output_nhdr
            process                = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error          = process.communicate()
            output_mask.append(output_nhdr)
    else:
        image_space                = nib.load(b0_normalized)
        predict                    = np.load(cases_mask_arr)
        predict[predict < 1]       = 0
        image_predict              = nib.Nifti1Image(predict, image_space.affine, image_space.header)
        output_file                = cases_mask_arr[:len(cases_mask_arr) - len(suffix_npy)] + 'nii.gz'
        nib.save(image_predict, output_file)
        downsample_file            = output_file[:len(output_file) - (len(output_file)+1)] + '-downsampled.nii.gz'
        bashCommand_downsample     = "ResampleImage 3 " + output_file + " " + downsample_file + " " + dim[0] + "x" + dim[1] + "x" + dim[2] + " 1"
        output2                    = subprocess.check_output(bashCommand_downsample, shell=True)
        output_mask                = sub_name[:len(sub_name) - (len(suffix_nhdr)+1)] + '_BrainMask.nhdr'
        bashCommand                = 'ConvertBetweenFileFormats ' + downsample_file + " " + output_nhdr
        process                    = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error              = process.communicate()
        
    return output_mask

def clear():
    print "Cleaning files ..."
    for filename in os.listdir(os.getcwd()):
        if filename.startswith('dwi') | filename.endswith(suffix_npy) | filename.endswith(suffix_nifti_gz):
            os.unlink(filename)

def split(cases_file, split_dim, case_arr):
    '''
    Parameters
    ---------
    cases_file : str
                 Accepts a filename which is in 3d numpy array format stored in disk
    split_dim  : list
                 Contains the "x" dim for all the cases
    case_arr   : list
                 Contain filename for all the input cases

    Returns
    --------
    predict_mask : list
                   Contains the predicted mask filename of all the cases which is stored in disk in *.npy format
    '''
    count           = 0
    start           = 0
    end             = start + split_dim[0]
    SO              = np.load(cases_file)
    predict_mask    = [] 
    for i in range(0, len(split_dim)):
        end         = start + split_dim[i]
        casex       = SO[start:end,:,:]
        input_file  = str(case_arr[i])
        output_file = input_file[:len(input_file) - (len(suffix_nhdr)+1)] + '_SO.npy'
        predict_mask.append(output_file)
        np.save(output_file, casex)
        start       = end
        count      += 1
    
    return predict_mask


if __name__ == '__main__':
    # check if file exists

    if args.dwi:
        f = pathlib.Path(args.dwi)
        if f.exists():
            print "File exist"
            filename = os.path.basename(args.dwi)
        else:
            print "File not found"
            sys.exit(1)

        # Input caselist.txt
        if filename.endswith(suffix_txt):
            with open(filename) as f:
                case_arr = f.read().splitlines()
        
            binary_file            = 'binary'
            f_handle               = open(binary_file, 'wb')
            x_dim                  = 0
            y_dim                  = 256
            z_dim                  = 256
            split_dim              = []
            b0_normalized_cases    = []
            cases_dim              = []
            for subjects in case_arr:
                input_file         = subjects
                f                  = pathlib.Path(input_file)
                if f.exists():
                    check_gradient(input_file)
                    b0_nhdr        = extract_b0(input_file)
                    b0_nii         = nhdr_to_nifti(b0_nhdr)
                    dimensions     = get_dimension(b0_nii)
                    cases_dim.append(dimensions)
                    x_dim         += int(dimensions[0])
                    split_dim.append(int(dimensions[0]))  
                    b0_resampled   = resample(b0_nii, dimensions[0])
                    b0_normalized  = normalize(b0_resampled)
                    b0_normalized_cases.append(b0_normalized)
                    img            = nib.load(b0_normalized)
                    imgU16         = img.get_data().astype(np.float32)
                    imgU16.tofile(f_handle)
                else:
                    print "File not found ", input_file 
                    sys.exit(1)
            f_handle.close()
            print "Merging npy files"
            cases_file = 'casefile.npy'
            merge = np.memmap(binary_file, dtype=np.float32, mode='r+', shape=(x_dim, y_dim, z_dim))
            print "Saving training data to disk"
            np.save(cases_file, merge)
            dwi_mask_npy = predict_mask(cases_file)
            cases_mask_arr = split(dwi_mask_npy, split_dim, case_arr)
            dwi_mask = npy_to_nhdr(b0_normalized_cases, cases_mask_arr, case_arr, cases_dim)
            for masks in dwi_mask:
                print "Mask file = ", masks
            clear()

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
