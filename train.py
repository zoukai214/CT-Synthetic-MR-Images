from unet_model import *
from keras.callbacks import TensorBoard

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def train_and_predict():
    print("-" * 30)
    print("Loading and preprocessing train data ...")
    print("-" * 30)
    
    imgs_train = np.load('/home/zoukai/project/unet-master/unet-master/data/input_npy/7_trainct.npy')
    imgs_mask_train = np.load('/home/zoukai/project/unet-master/unet-master/data/input_npy/7_trainmri.npy')

    #imgs_train = np.reshape(imgs_train,imgs_train.shape+(1,))
    #imgs_mask_train = np.reshape(imgs_mask_train,imgs_mask_train.shape+(1,))

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    total = imgs_train.shape[0]
    print("-" * 30)
    print("Create and compiling model...")
    print('-' * 30)
    model = unet()
    model_checkpoint = ModelCheckpoint('7_normalization.hdf5', monitor='loss', verbose=1, save_best_only=True)
    print("-" * 30)
    print("Fitting model ...")
    print("-" * 30)

    tb = TensorBoard(log_dir='./logs',  # log 目录
                     histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=32,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)
    

    model.fit(imgs_train, imgs_mask_train, batch_size=16, epochs=1000,validation_split=0.2, verbose=1, shuffle=True,
              callbacks=[model_checkpoint,tb])

if __name__ == "__main__":
    train_and_predict()
