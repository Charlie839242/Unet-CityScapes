import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import tensorflow as tf
import glob
import pandas as pd
from datasets import DataLoader, DataDecoder
from model import UNET, MeanIoU, UpdatedMeanIoU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    # 1. Decode dataset
    if (len(glob.glob('./CityScapes/train/rgb/*.png')) == 0):
        DataDecoder = DataDecoder('./CityScapes/Lab2_train_data.h5', './CityScapes/Lab2_test_data.h5')
        DataDecoder.decode_train_rgb()
        DataDecoder.decode_train_seg()
        DataDecoder.decode_test_rgb()
        DataDecoder.decode_test_seg()
        print('PNG images are extracted successfully')
    else:
        print('PNG images already extracted')

    # 2. Load dataset
    DataLoader = DataLoader('./CityScapes/train/rgb/*.png', './CityScapes/train/seg/*.png',
                            './CityScapes/test/rgb/*.png', './CityScapes/test/seg/*.png')

    BATCH_SIZE = 1
    AUTO = tf.data.experimental.AUTOTUNE  # automatic load
    # load train set
    train_dataset_path = tf.data.Dataset.from_tensor_slices((DataLoader.train_rgb_path, DataLoader.train_seg_path))
    train_dataset = train_dataset_path.map(DataLoader.load_trainset, num_parallel_calls=AUTO)
    train_dataset = train_dataset.repeat().batch(BATCH_SIZE).prefetch(AUTO)
    # load test set
    test_dataset_path = tf.data.Dataset.from_tensor_slices((DataLoader.test_rgb_path, DataLoader.test_seg_path))
    test_dataset = test_dataset_path.map(DataLoader.load_testset, num_parallel_calls=AUTO)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    print('Training Dataset:\n', train_dataset)

    # 3. Reload the model

    model = UNET()
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc', UpdatedMeanIoU(num_classes=34)])

    weights = tf.train.latest_checkpoint('./models')
    model.load_weights(weights)

    # 4. Training
    EPOCHS = 80
    model_path = './models/model.ckpt'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                model_path, verbose=1, save_weights_only=True, period=1)

    train_step = DataLoader.num_train // BATCH_SIZE
    val_step = DataLoader.num_test // BATCH_SIZE

    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        steps_per_epoch=train_step,
                        validation_data=test_dataset,
                        validation_steps=val_step,
                        callbacks=model_checkpoint_callback)

    # save history
    pd.DataFrame(history.history).to_csv('training_log.csv', index=False)





