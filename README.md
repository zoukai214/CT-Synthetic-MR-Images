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
![改进的unet](https://www.google.com/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwivnLmZyfHcAhVpFjQIHQ1bDboQjRx6BAgBEAU&url=https%3A%2F%2Flink.springer.com%2Fchapter%2F10.1007%2F978-3-319-67389-9_34&psig=AOvVaw1Cyn7XQyG7UwXM33T3AGlm&ust=1534508883025472)
参考unet_model.py文件

## 训练
train.py
