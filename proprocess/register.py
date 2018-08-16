import os
import SimpleITK as sitk

def rig_register(CT_DIR ,MRI_DIR,OUTPUT_DIR,OUTPUT_NAME):

    # fixed_image =  sitk.ReadImage(fdata("training_001_ct.mha"), sitk.sitkFloat32)
    fixed_image =  sitk.ReadImage(CT_DIR, sitk.sitkFloat32)
    # 取DWI数据，即取mri中的前20层
    fixed_image = fixed_image[:, :, :20]
    moving_image = sitk.ReadImage(MRI_DIR, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    sitk.WriteImage(moving_resampled, os.path.join(OUTPUT_DIR, '{}.nii'.format(OUTPUT_NAME)))
    sitk.WriteTransform(final_transform, os.path.join(OUTPUT_DIR, '{}.tfm'.format(OUTPUT_NAME)))

# if __name__ == '__main__':
#     #输入ct文件的路径
#     CT_DIR = "/home/zoukai/Data/CTMR/CT/20180703_1704151HeadRoutines004a002.nii"
#     #输入mri文件的路径
#     MRI_DIR = "/home/zoukai/Data/CTMR/MR/sub-25634_ses-1_T1w.nii"
#     #输出文件夹的路径
#     OUTPUT_DIR = "/home/zoukai/Data/CTMR/MR_2/"
#     #输出文件的名字
#     OUTPUT_NAME = "ZK_007"
#     rig_register(CT_DIR, MRI_DIR,OUTPUT_DIR, OUTPUT_NAME)
