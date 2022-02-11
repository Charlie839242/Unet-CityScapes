import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
from model import UNET, UpdatedMeanIoU
from datasets import DataLoader
import tensorflow as tf


# Load model and weights
model = UNET()
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['acc', UpdatedMeanIoU(num_classes=34)])
weights = tf.train.latest_checkpoint('./models')
model.load_weights(weights).expect_partial()


# Load test set
DataLoader = DataLoader('./CityScapes/train/rgb/*.png', './CityScapes/train/seg/*.png',
                        './CityScapes/test/rgb/*.png', './CityScapes/test/seg/*.png')

BATCH_SIZE = 1
AUTO = tf.data.experimental.AUTOTUNE  # automatic load

test_dataset_path = tf.data.Dataset.from_tensor_slices((DataLoader.test_rgb_path, DataLoader.test_seg_path))
test_dataset = test_dataset_path.map(DataLoader.load_testset, num_parallel_calls=AUTO)
test_dataset = test_dataset.batch(BATCH_SIZE)
print('Test Dataset:\n', test_dataset)


# evaluation
model.evaluate(test_dataset)

