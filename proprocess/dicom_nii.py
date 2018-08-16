#!/usr/bin/python
# -*- coding:utf-8 -*-
import SimpleITK as sitk
import os
import numpy as np

def ReadDicom(root,OUTPUT_DIR):
    for filename in os.listdir(root):
        pathname = os.path.join(root,filename)
        if (os.path.isdir(pathname)):
            for two_file in os.listdir(pathname):
                two_file_dir = os.path.join(pathname,two_file)
                if (os.path.isdir(two_file_dir)):
                    if two_file == "CT":
                        two_pathname = os.path.join(two_file_dir,"DICOM")
                        #print(two_file_dir)
                        reader = sitk.ImageSeriesReader()
                        dicom_names = reader.GetGDCMSeriesFileNames(two_pathname)
                        reader.SetFileNames(dicom_names)
                        image = reader.Execute()
                        sitk.WriteImage(image, os.path.join(OUTPUT_DIR, '{}_{}.nii'.format(filename,two_file)))

def regieter(input_dir,OUTPUT_DIR):
    L = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            num = file.rfind('_')
            ct_name = file[num+1:]
            mri_name = file[:num]
            if ct_name == "CT.nii":
                zk_mri_path = mri_name+'_mri.nii'
                mri_path = os.path.join(root,zk_mri_path)
                if os.path.exists(mri_path):
                    MRI_DIR = mri_path
                else:
                    MRI_DIR = mri_name +'_DWI.nii'
                    MRI_DIR = os.path.join(root,MRI_DIR)

                CT_DIR = os.path.join(root,file)
                print(CT_DIR,MRI_DIR)
                # fixed_image =  sitk.ReadImage(fdata("training_001_ct.mha"), sitk.sitkFloat32)
                fixed_image = sitk.ReadImage(MRI_DIR, sitk.sitkFloat32)
                ## 取其中的体素spacing
                fixed_image_spaceing = fixed_image.GetSpacing()
                moving_image = sitk.ReadImage(CT_DIR, sitk.sitkFloat32)
                # 取DWI数据，即取mri中的前20层
                fixed_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed_image)[0])
                fixed_image.SetSpacing(fixed_image_spaceing)



                initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                                      moving_image,
                                                                      sitk.Euler3DTransform(),
                                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)


                registration_method = sitk.ImageRegistrationMethod()

                # Similarity metric settings.
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
                registration_method.SetMetricSamplingPercentage(0.01)

                registration_method.SetInterpolator(sitk.sitkLinear)

                # Optimizer settings.
                registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                                  convergenceMinimumValue=1e-6,
                                                                  convergenceWindowSize=10)
                registration_method.SetOptimizerScalesFromPhysicalShift()

                # Setup for the multi-resolution framework.

                registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
                registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
                registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

                # Don't optimize in-place, we would possibly like to run this cell multiple times.
                registration_method.SetInitialTransform(initial_transform, inPlace=False)
                final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                              sitk.Cast(moving_image, sitk.sitkFloat32))
                moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                                 moving_image.GetPixelID())

                sitk.WriteImage(moving_resampled, os.path.join(OUTPUT_DIR, 'sys_{}_mri.nii'.format(mri_name)))
                #sitk.WriteTransform(final_transform, os.path.join(OUTPUT_DIR, 'sys_{}.tfm'.format(mri_name)))


def sys_mri(input_dir,OUTPUT_DIR):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            num = file.rfind('_')
            ct_name = file[num+1:]
            mri_name = file[:num]
            if ct_name == "CT.nii":
                zk_mri_path = mri_name+'_mri.nii'
                mri_path = os.path.join(root,zk_mri_path)
                if os.path.exists(mri_path):
                    MRI_DIR = mri_path
                else:
                    MRI_DIR = mri_name +'_DWI.nii'
                    MRI_DIR = os.path.join(root,MRI_DIR)
                MRI_IMAGE = sitk.ReadImage(MRI_DIR, sitk.sitkFloat32)
                mri_image_spaceing = MRI_IMAGE.GetSpacing()
                mri_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(MRI_IMAGE)[0])
                mri_image.SetSpacing(mri_image_spaceing)
                sitk.WriteImage(mri_image, os.path.join(OUTPUT_DIR, 'sys_{}_mri.nii'.format(mri_name)))
#root = "/media/zoukai/软件/yikai/Brain_MRI2_nanshan/"
#OUTPUT_DIR = "/home/zoukai/Data/CTMR/DWI2"
#ReadDicom(root,OUTPUT_DIR)
input_dir = "/media/zoukai/软件/DATA/Brain_MRI2/"
OUTPUT_DIR = "/media/zoukai/软件/DATA/Brain_3D_MRI/"
sys_mri(input_dir,OUTPUT_DIR)
#regieter(input_dir,OUTPUT_DIR)





