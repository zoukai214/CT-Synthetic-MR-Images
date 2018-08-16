#合成128*128图像npy，作为训练数据/测试数据
import glob
import os
import numpy as np
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
%matplotlib inline
image_path = "/home/zoukai/Data/CTMR/dwicrop/8_trainct"
#1.读取文件夹中的文件
image_name_arr = glob.glob(os.path.join(image_path,"*.nii"))

#2.预设合成npy的维度与步长
imgs = np.ndarray((5120,128,128,1),dtype = np.uint8)
output_shape = np.array([128,128,1])
stride = np.array([128,128,1])

#3.遍历每一个文件，并取出其中128x128
c = 0
for index,item in enumerate(image_name_arr):
    mri_image = sitk.ReadImage(item,sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(mri_image).astype(np.float32)
    img = img.transpose(2,1,0)
    # ct数据进行加1000的处理
    img = img+1000.
    
    #3.判断是否是192x192分辨率的
    if img.shape[0] == 192:
        for i in range(img.shape[2]):
            for h in range(2):
                for w in range(2):
                    image_path = img[h*(stride[0]-64):h*(stride[0]-64)+output_shape[0],
                                 w*(stride[1]-64):w*(stride[1]-64)+output_shape[1],
                                 i*stride[2]:i*stride[2]+output_shape[2]]
                    imgs[c] = image_path
                    c += 1
                    print(c)
    else:
        for i in range(img.shape[2]):
            image_path = img[:,:,i*stride[2]:i*stride[2]+output_shape[2]]
            imgs[c] = image_path
            c+=1
            print(c)
           
np.save('../data/input_npy/8_trainct.npy',imgs)
