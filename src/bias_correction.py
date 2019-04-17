from nipype.interfaces.ants import N4BiasFieldCorrection

def correct_bias(in_file, out_file):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. 
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    done = correct.run()
    return done.outputs.output_image

    
cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/caselistfinal.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

count = 0
for subject in case_arr:
    in_file = subject + '/dwib0.nii.gz'
    out_file = subject + '/dwib0_biasCorrected.nii.gz'
    correct_bias(in_file, out_file)
    print("Case ", count, "done")
    count = count + 1
