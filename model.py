import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def UNET():
    input = keras.Input(shape=(256,256,3))
    x1 = layers.Conv2D(64,3,strides=1,padding='same')(input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(64,3,strides=1,padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)              # (256,256,64)

    x2 = layers.MaxPooling2D()(x1)      # (128,128,64)

    x2 = layers.Conv2D(128,3,strides=1,padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(128,3,strides=1,padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)              # (128,128,128)

    x3 = layers.MaxPooling2D()(x2)      # (64,64,128)

    x3 = layers.Conv2D(256,3,strides=1,padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(256,3,strides=1,padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)              # (64,64,256)

    x4 = layers.MaxPooling2D()(x3)      # (32,32,256)

    x4 = layers.Conv2D(512,3,padding='same',strides=1)(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    x4 = layers.Conv2D(512,3,padding='same',strides=1)(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)              # (32,32,512)

    x5 = layers.MaxPooling2D()(x4)      # (16,16,512)

    x5 = layers.Conv2D(1024,3,padding='same',strides=1)(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.ReLU()(x5)
    x5 = layers.Conv2D(1024,3,padding='same',strides=1)(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.ReLU()(x5)              # (16,16,1024)

    x4_ = layers.Conv2DTranspose(512,2,padding='same',strides=2)(x5)
    x4_ = layers.BatchNormalization()(x4_)
    x4_ = layers.ReLU()(x4_)            # (32,32,512)

    x3_ = tf.concat([x4,x4_],axis=-1)   # (32,32,1024)
    x3_ = layers.Conv2D(512,3,padding='same',strides=1)(x3_)
    x3_ = layers.BatchNormalization()(x3_)
    x3_ = layers.ReLU()(x3_)
    x3_ = layers.Conv2D(512,3,padding='same',strides=1)(x3_)
    x3_ = layers.BatchNormalization()(x3_)
    x3_ = layers.ReLU()(x3_)            # (32,32,512)

    x3_= layers.Conv2DTranspose(256,2,padding='same',strides=2)(x3_)
    x3_ = layers.BatchNormalization()(x3_)
    x3_ = layers.ReLU()(x3_)            # (64,64,256)

    x2_ = tf.concat([x3,x3_],axis=-1)   # (64,64,512)
    x2_ = layers.Conv2D(256,3,padding='same',strides=1)(x2_)
    x2_ = layers.BatchNormalization()(x2_)
    x2_ = layers.ReLU()(x2_)
    x2_ = layers.Conv2D(256,3,padding='same',strides=1)(x2_)
    x2_ = layers.BatchNormalization()(x2_)
    x2_ = layers.ReLU()(x2_)            # (64,64,256)

    x2_ = layers.Conv2DTranspose(128,2,padding='same',strides=2)(x2_)
    x2_ = layers.BatchNormalization()(x2_)
    x2_ = layers.ReLU()(x2_)            # (128,128,128)

    x1_ = tf.concat([x2, x2_], axis=-1)   # (128,128,256)
    x1_ = layers.Conv2D(128, 3, padding='same', strides=1)(x1_)
    x1_ = layers.BatchNormalization()(x1_)
    x1_ = layers.ReLU()(x1_)
    x1_ = layers.Conv2D(128, 3, padding='same', strides=1)(x1_)
    x1_ = layers.BatchNormalization()(x1_)
    x1_ = layers.ReLU()(x1_)            # (128,128,128)

    x1_ = layers.Conv2DTranspose(64,2,padding='same', strides=2)(x1_)
    x1_ = layers.BatchNormalization()(x1_)
    x1_ = layers.ReLU()(x1_)            # (256,256,64)

    x_ = tf.concat([x1,x1_],axis=-1)    # (256,256,128)
    x_ = layers.Conv2D(64,3,padding='same',strides=1)(x_)
    x_ = layers.BatchNormalization()(x_)
    x_ = layers.ReLU()(x_)
    x_ = layers.Conv2D(64,3,padding='same',strides=1)(x_)
    x_ = layers.BatchNormalization()(x_)
    x_ = layers.ReLU()(x_)              # (256,256,64)

    # output 34 classes
    output = layers.Conv2D(34, 1, padding='same', strides=1, activation='softmax')(x_)  # (256,256,34)

    return keras.Model(inputs=input, outputs=output)


'''
For the label image, there are two ways of coding: One Hot Coding and Sequential Coding.
The model output is always One Hot Coding.
When our labels are Sequential Coding, then loss function should be 'categorical_crossentropy'.
When our labels are One Hot Coding, then loss function should be 'sparse_categorical_crossentropy'.
When our labels are One Hot Coding, for metrics we can directly use 'tf.keras.metrics.MeanIoU'.
When our labels are Sequential Coding, we have to modify 'tf.keras.metrics.MeanIoU' a little as below.
MeanIoU is for tensorflow 2.0.
UpdateMeanIoU is for tensorflow 2.3 or higher.
'''


class MeanIoU(tf.keras.metrics.MeanIoU):
   def __call__(self, y_true, y_pred, sample_weight=None):
       y_pred = tf.argmax(y_pred, axis=-1)
       return super().__call__(y_true, y_pred, sample_weight=sample_weight)


class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, y_true=None, y_pred=None, num_classes=None, name=None, dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


if __name__ == '__main__':
    model = UNET()
    model.summary()




