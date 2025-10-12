import nilearn.image as ni
from sklearn.feature_extraction import img_to_graph
import numpy as np
import os
import SimpleITK as sitk
import glob

subdir = '/deneb_disk/disc_mri/scan_3_20_2024/nifti' #'/deneb_disk/Lung_Volumes/nifti_files/Svr106/'
outsubdir = '/deneb_disk/disc_mri/scan_3_20_2024/nifti_rot' #'/deneb_disk/Lung_Volumes/lung_stacks/Svr106_rot'

if not os.path.isdir(outsubdir):
    os.makedirs(outsubdir)

sub_files = glob.glob(subdir+'/*head*.nii.gz')

for s in sub_files:

    sub_file=os.path.basename(s)

    sub_img = os.path.join(subdir,sub_file)
    out_file = os.path.join(outsubdir,'p' + sub_file)


    img = sitk.ReadImage(sub_img)
    print (img.GetSpacing())

    sliceaxis = np.argmax(img.GetSpacing())

    if sliceaxis == 2:
        img2 = img

    if sliceaxis == 1:
        img2 = sitk.PermuteAxes(img, [2,0,1])

    if sliceaxis == 0:
        img2 = sitk.PermuteAxes(img, [1,2,0])


    print(img2.GetSpacing())

    sitk.WriteImage(img2, out_file)

