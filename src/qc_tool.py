import nibabel as nib


def get_header_info(header):    
    dim = header['dim']
    space_oriX = round(float(header['qoffset_x']), 2)
    space_oriY = round(float(header['qoffset_y']), 2)
    space_oriZ = round(float(header['qoffset_z']), 2)
    space_origin = [space_oriX, space_oriY, space_oriZ]
    space_dirX = header['srow_x']
    space_dirY = header['srow_y']
    space_dirZ = header['srow_z']
    return(list(dim[1:4]), space_origin, list(space_dirX[0:3]), 
                       list(space_dirY[0:3]), list(space_dirZ[0:3]))

def print_header(header):
    print("Dimension = ", header[0])
    print("Space Origin = ", header[1])
    print(header[2])
    print(header[3])
    print(header[4])
    
def check_header(A, B, subject):
   
   if A[0] != B[0]:  
       print ("The dimensions are not identical")     
   if A[1] != B[1]: 
       print ("The Space Origin are not identical")  
       with open("origin.txt", "a") as myfile:
           myfile.write(subject + "\n")
   if A[2] != B[2] or A[3] != B[3] or A[4] != B[4]:  
       print ("The Space directions are not identical")
    
cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/original/caselistfinal.txt'

with open(cases) as f:
    case_arr = f.read().splitlines()
    
for subject in case_arr:
    img_dwi = nib.load(subject+ '/dwib0.nii.gz')
    img_mask = nib.load(subject+ '/truth.nii.gz')
    dwi_header = img_dwi.header
    mask_header = img_mask.header
    dwi_header_details = get_header_info(dwi_header)
    mask_header_details = get_header_info(mask_header)
    #print(dwi_header)
    #print(mask_header)
    #print(subject)
    #print()
    #print("DWI Header Info")
    #print_header(dwi_header_details)
    #print()
    #print("DWI Mask Info")
    #print_header(mask_header_details)
    #print()
    check_header(dwi_header_details, mask_header_details, subject)
    print("-------------------------------------------------")
