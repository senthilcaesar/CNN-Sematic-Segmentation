import numpy as np
import nibabel as nib


def print_header(case_arr):
    dim1_total = 0
    for subject in case_arr:

        img = nib.load(subject + '/' + 'dwib0-linear-99percentile-pad.nii.gz')
        header = img.header
        dim = header['dim']
        addition = int(dim[1])
        dim1_total += addition
        print(dim1_total)

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/brats/goodMasks_shuffled.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

print_header(case_arr)
