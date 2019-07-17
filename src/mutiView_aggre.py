import numpy as np

predict_sagittal = np.load('AP107-predict.npy')
predict_coronal = np.load('PA107-predict.npy')
predict_axial = np.load('AP99-predict.npy')

m,n = predict_sagittal.shape[::2]
x = predict_sagittal.transpose(0,3,1,2).reshape(m,-1,n)

m,n = predict_coronal.shape[::2]
y = predict_coronal.transpose(0,3,1,2).reshape(m,-1,n)

m,n = predict_axial.shape[::2]
z = predict_axial.transpose(0,3,1,2).reshape(m,-1,n)

sagittal_view = list(x.ravel())
coronal_view = list(y.ravel())
axial_view = list(z.ravel())


sagittal = []
coronal = []
axial = []

for i in range(0, len(sagittal_view)):
    
    vector_sagittal = [1 - sagittal_view[i], sagittal_view[i]]
    vector_coronal = [1 - coronal_view[i], coronal_view[i]]
    vector_axial = [1 - axial_view[i], axial_view[i]]
    
    sagittal.append(np.array(vector_sagittal))
    coronal.append(np.array(vector_coronal))
    axial.append(np.array(vector_axial))
    

prob_vector = []

for i in range(0, len(sagittal_view)):
    
    val = np.argmax(0.4*coronal[i] + 0.4*axial[i] + 0.2*sagittal[i])
    prob_vector.append(val)
    
data = np.array(prob_vector)
shape = (140, 256, 256)
brain_mask = data.reshape(shape)
    
    
import nibabel as nib

subject_label = 'AP107-predict.nii'
data_label = nib.load(subject_label)

image_predict = nib.Nifti1Image(brain_mask, data_label.affine, data_label.header)
nib.save(image_predict, 'predict-mask.nii')
