import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

para_type = 'float32'
L2_WEIGHT_DECAY = 0.02
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 0.001

def FPNModel(num_classes, input_shape):

    input_data = keras.layers.Input(shape=input_shape, dtype = para_type)
    x = input_data
    bn_axis = 4

    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='valid', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)

    '''initial down sampling if not enough GPU memory '''
    #x = keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), dtype = para_type)(x)

    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='valid', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    l1 = keras.layers.GlobalAveragePooling3D(dtype = para_type)(x)
    x = keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), dtype = para_type)(x)

    x = keras.layers.Dropout(0.2,dtype = para_type)(x)

    shortcut = x
    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    x = x + shortcut
    l2 = keras.layers.GlobalAveragePooling3D(dtype = para_type)(x)
    x = keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), dtype = para_type)(x)

    shortcut = x
    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    x = x + shortcut
    l3 =  keras.layers.GlobalAveragePooling3D(dtype = para_type)(x)
    x = keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), dtype = para_type)(x)
    x = keras.layers.Dropout(0.2,dtype = para_type)(x)


    shortcut = x
    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    x = x + shortcut
    l4 = keras.layers.GlobalAveragePooling3D(dtype = para_type)(x)
    x = keras.layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), dtype = para_type)(x)

    shortcut = x
    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    x = keras.layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', dtype = para_type)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, dtype = para_type)(x)
    x = keras.layers.ReLU(dtype = para_type)(x)
    x = x + shortcut
    l5 = keras.layers.GlobalAveragePooling3D(dtype = para_type)(x)

    x = keras.layers.concatenate([l1,l2,l3,l4,l5], axis = 1, dtype = para_type)

    x = keras.layers.Dropout(0.5,dtype = para_type)(x)
    x = keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),bias_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),dtype = para_type)(x)


    x = keras.layers.Dense(
        num_classes, activation='softmax',dtype='float32',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(x)

    return keras.models.Model([input_data], x, name='FPNModel')


def AAFusion(num_classes):

    ''' image input'''
    x1 = keras.layers.Input(shape=(160))
    ''' clinical data input'''
    x2 = keras.layers.Input(shape=(1))
    x3 = keras.layers.Input(shape=(1))
    x4 = keras.layers.Input(shape=(1))
    x5 = keras.layers.Input(shape=(4))
    x6 = keras.layers.Input(shape=(4))

    x = keras.layers.concatenate([x1,x2,x3,x4,x5,x6])
    att = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=171,attention_axes=1)
    x, w = att(x, x, return_attention_scores = True)

    bn_axis = 0
    x = keras.layers.Dropout(0.5,dtype = para_type)(x)
    x = keras.layers.Dense(32, activation='relu',kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),bias_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),dtype = para_type)(x)
    x = keras.layers.Dense(
        num_classes, activation='softmax',dtype='float32',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(x)

    return keras.models.Model([x1,x2,x3,x4,x5,x6], x)

# Create model
FPN = FPNModel(2, (157, 213, 217,1))
AD_MCI_fusion = AAFusion(2)
