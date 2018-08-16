import SimpleITK as sitk
import os

def N4(input_dir,OUTPUT_DIR):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            num = file.rfind('_')
            ct_label = file[num + 1:]
            if ct_label == "mri.nii":
                ct_path = os.path.join(root, file)
                print(ct_path)
                inputImage = sitk.ReadImage(ct_path)
                maskImage = sitk.OtsuThreshold(inputImage,0,1,200)

                inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)

                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                output = corrector.Execute(inputImage,maskImage)
                sitk.WriteImage(output, os.path.join(OUTPUT_DIR, '{}.nii'.format(file)))
input_dir = "/media/zoukai/软件/DATA/Brain_register_ct/"
OUTPUT_DIR = "/media/zoukai/软件/DATA/Brain_N4/"
N4(input_dir, OUTPUT_DIR)