import numpy as np
import SimpleITK as sitk

data = [sitk.ReadImage('/autofs/space/dura_002/users/sq566/HCP/110007/110007-dwib0.nhdr')]
dimension = data[0].GetDimension()

reference_physical_size = np.zeros(dimension)
for img in data:
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    

reference_origin = np.zeros(dimension)
reference_direction = np.identity(dimension).flatten()
reference_size = [144, 96, 145] #*dimension # Arbitrary sizes, smallest size that yields desired results. 
reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

reference_image = sitk.Image(reference_size, data[0].GetPixelIDValue())
reference_image.SetOrigin(reference_origin)
reference_image.SetSpacing(reference_spacing)
reference_image.SetDirection(reference_direction)

reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

for img in data:
    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    #transform = sitk.AffineTransform(dimension)
    transform = sitk.VersorTransform((0,0,1), np.pi)
    transform.SetMatrix(img.GetDirection())
    #transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    sitk.Show(sitk.Resample(img, reference_image, centered_transform, sitk.sitkBSpline, 0.0), debugOn=True)
    
