import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
# from keras.layers import *
from keras.layers import MaxPooling2D, UpSampling2D, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from keras.layers import BatchNormalization


def my_loss(y_true, y_pred):

    m,n_H,n_W,n_C = y_pred.get_shape().as_list()
    y_true_unrolled = keras.transpose(keras.reshape(y_true,[-1]))
    y_pred_unrolled = keras.transpose(keras.reshape(y_pred,[-1]))
    J_content = keras.sum(keras.square(y_pred-y_true)/(4*n_H*n_W*n_C))
    return J_content

def unet(pretrained_weights=None, input_size=(128, 128, 1)):
    inputs = Input(input_size)
    # conv1 = model.add(Conv2D(64,3,activation='relu',padding='same',kernel_initial='he_normal')(inputs))
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    #drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    #drop5 = Dropout(0.2)(conv5)

    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)
    up6 = UpSampling2D((2, 2))(conv5)
    # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)

    # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)
    up7 = UpSampling2D((2,2))(conv6)
    # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)

    # up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)
    up8 = UpSampling2D((2,2))(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=-1)(conv8)

    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)
    up9 = UpSampling2D((2,2))(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis=-1)(conv9)

    merge10 = concatenate([inputs,conv9],axis=3)
    #conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = Conv2D(1, 1, activation='relu',padding='same')(merge10)

    model = Model(input=inputs, output=conv10)

    #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy','mae'])
    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
