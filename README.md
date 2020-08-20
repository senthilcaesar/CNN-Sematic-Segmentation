Note!

Please use the below latest version of our work.

https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation

## Getting Started

### Pre-requisites

You need to have following in order for this library to work as expected

01)  python 3.6
02)  pip >= 19.0
03)  numpy >= 1.16.4
04)  nibabel >= 2.2.1
05)  tensorflow-gpu >= 1.12.0
06)  keras >= 2.2.4
07)  cudatoolkit = 9.0
08)  cudnn = 7.0.5

### Python 3

Download [Miniconda Python 3.6 bash installer](https://docs.conda.io/en/latest/miniconda.html) (32/64-bit based on your environment):
    
    sh Miniconda3-latest-Linux-x86_64.sh -b # -b flag is for license agreement

Activate the conda environment:

    source ~/miniconda3/bin/activate # should introduce '(base)' in front of each line
    
### Install prerequisites for running the pipeline

#### For CPU
01) conda install cudatoolkit=9.0
02) conda install cudnn=7.0.5
03) pip install tensorflow==1.12.0

#### For GPU
01) conda install cudatoolkit=9.0
02) conda install cudnn=7.0.5
03) pip install tensorflow-gpu==1.12.0

#### For Either CPU or GPU
01) conda install -c pnlbwh ants
02) pip install keras==2.2.4
03) pip install nibabel
14) pip install gputil

### Setting CUDA Path
The NVIDIA graphics driver and CUDA compilier are already installed on machines that support CUDA. However, one must set environment variables in order to run and write CUDA enabled programs.

If you use bash, add the following lines to the bottom of your .bashrc file:

        # add cuda tools to command path
        export PATH=/usr/local/cuda/bin:${PATH}

        # add the CUDA binary and library directory to your LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
  
Log out and back in for the changes to take effect.

### Download model architecture, weights and IIT mean b0 template

Download the following data and place them under `model_folder/` directory
```
pip install gdown
gdown --id 15dJ-ZpRznTlcU6h4LLGUOSPSH6qY1xbg --output trainedmodel.zip
gdown https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz
mkdir model_folder
mv trainedmodel.zip model_folder/
cd model_folder
unzip trainedmodel.zip
```

### Running the pipeline

```
python dwi_masking.py -i cases.txt -f model_folder
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Note: cases.txt should contain the full path to the diffusion volumes
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/home/pycharm/data/compnet/subject01/subject01_dwi.nii.gz

## Model Architecture
The code is written by Raunak Dey available at 

https://github.com/raun1/MICCAI2018---Complementary_Segmentation_Network-Raw-Code. 

In summary, his proposed architecture is designed in the framework of encoder-decoder networks and have three pathways.

* Segmentation Branch - learns what is the brain tissue and to generate a brain mask 

* Complementary Branch - learns what is outside of the brain and to help the other
branch generate brain mask

* Reconstruction Branch - It provides direct feedback to the segmentation and
complementary branche and expects reasonable predictions from them as input to reconstruct the original input image.
![Screenshot](https://github.com/SenthilCaesar/CNN-Brain-MRI-Segmentation/blob/master/CompNet%20Arch.png)


## Multi View Aggregation step:
> The approach is to train 3 separate networks for three principal axes ( Sagittal, Coronal and axial ) and 
to perform multi-view aggregation step that combines segmentations from models trained on 2D slices along three principal axes: coronal, sagittal and axial. The final segmentation would be obtained by combining the probability maps from all three segmentation.
![Screenshot](https://github.com/SenthilCaesar/CNN-Brain-MRI-Segmentation/blob/master/Multiview.png)
