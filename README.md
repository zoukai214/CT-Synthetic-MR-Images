# ct_to_mri
# input ct data use  U-net method systh mri
来自论文《Whole Brain Segmentation and Labeling from CT Using Synthetic MR Images》，主要用于通过分割模型，将CT数据合成MRI数据。

## 运行环境
   keras>1.3
   Tensorfow>1.0
   SimppleITK=1.1.0
## 数据收集
收集同一人的CT/MRI数据，需要对数据进行N4偏差场矫正，白质均值正则化，再讲两个数据进行刚性配准。

## 模型的修改
![改进的unet](https://github.com/zoukai214/CT-Synthetic-MR-Images/tree/master/Screenshots/modifi_unet.png)

按上图对Unet进行修改，参考unet_model.py文件

## 数据预处理
![数据预处理](https://github.com/zoukai214/CT-Synthetic-MR-Images/tree/master/Screenshots/preprocession.png)
如上图对CT,MRI数据进行数据预处理
## 训练
train.py
