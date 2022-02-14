import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import h5py
import random
import numpy as np
import glob
import tensorflow as tf
from PIL import Image

# seed
random.seed(0)
np.random.seed(0)


class DataDecoder():
    def __init__(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset
        print('DataDecoder Initialized:\nEncoded train data: '+self.trainset+'\nEncoded test data: '+self.testset+'\n')

    def read_file(self, filepath):
        f = h5py.File(filepath, "r")
        color_codes, rgb, seg = f['color_codes'][:], f['rgb'][:], f['seg'][:]
        return f, color_codes, rgb, seg

    def decode_train_rgb(self):
        _, _, rgb_train, _ = self.read_file(self.trainset)
        num_train = rgb_train.shape[0]
        for i in range(num_train):
            image = Image.fromarray(rgb_train[i])
            image.save('./CityScapes/train/rgb/' + str(i) + '_rgb.png')

    def decode_train_seg(self):
        _, _, _, seg_train = self.read_file(self.trainset)
        num_train = seg_train.shape[0]
        for i in range(num_train):
            image = Image.fromarray(np.squeeze((seg_train[i])))     # You have to squeeze it !
            image.save('./CityScapes/train/seg/' + str(i) + '_seg.png')

    def decode_test_rgb(self):
        _, _, rgb_test, _ = self.read_file(self.testset)
        num_test = rgb_test.shape[0]
        for i in range(num_test):
            image = Image.fromarray(rgb_test[i])
            image.save('./CityScapes/test/rgb/' + str(i) + '_rgb.png')

    def decode_test_seg(self):
        _, _, _, seg_test = self.read_file(self.testset)
        num_test = seg_test.shape[0]
        for i in range(num_test):
            image = Image.fromarray(np.squeeze(seg_test[i]))
            image.save('./CityScapes/test/seg/' + str(i) + '_seg.png')


class DataLoader():
    def __init__(self, train_rgb_path, train_seg_path, test_rgb_path, test_seg_path):
        self.train_rgb_path = glob.glob(train_rgb_path)
        self.train_seg_path = glob.glob(train_seg_path)
        self.test_rgb_path = glob.glob(test_rgb_path)
        self.test_seg_path = glob.glob(test_seg_path)
        self.num_train = len(self.train_rgb_path)
        self.num_test = len(self.test_rgb_path)

        # To shuffle training dataset
        self.index = np.random.permutation(self.num_train)
        self.train_rgb_path = np.array(self.train_rgb_path)[self.index]
        self.train_seg_path = np.array(self.train_seg_path)[self.index]
        print('DataLoader Initialized:\nnum_train: '+str(self.num_train)+'\nnum_test: '+str(self.num_test)+'\n')

    def read_rgb(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        return image

    def read_seg(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=1)
        return image

    def resize(self, image, mask):      # random crop and resize to (256, 256)
        img = tf.concat([image, mask], axis=-1)
        img = tf.image.resize(img, (280, 280), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)   # only NEAREST_NEIGHBOR !
        img = tf.image.random_crop(img, size=[256, 256, 4])
        return img[:, :, :3], img[:, :, 3:]
        # image = tf.image.resize(image, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # mask = tf.image.resize(mask, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # return image, mask

    def normalize(self, image, mask):   # normalize to [-1, +1]
        image = tf.cast(image, tf.float32) / 127.5 - 1
        mask = tf.cast(mask, tf.int32)
        return image, mask

    def load_trainset(self, rgb_path, seg_path):            # load train set
        train_img = self.read_rgb(rgb_path)
        label_img = self.read_seg(seg_path)

        image, mask = self.resize(train_img, label_img)
        image, mask = self.normalize(image, mask)
        return image, mask

    def load_testset(self, rgb_path, seg_path):             # load test set
        train_img = self.read_rgb(rgb_path)
        label_img = self.read_seg(seg_path)

        train_img = tf.image.resize(train_img, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_img = tf.image.resize(label_img, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image, mask = self.normalize(train_img, label_img)
        return image, mask


if __name__ == '__main__':
    if len(glob.glob('./CityScapes/train/rgb/*.png')) == 0:
        DataDecoder = DataDecoder('./CityScapes/Lab2_train_data.h5', './CityScapes/Lab2_test_data.h5')
        DataDecoder.decode_train_rgb()
        DataDecoder.decode_train_seg()
        DataDecoder.decode_test_rgb()
        DataDecoder.decode_test_seg()
    else:
        print('PNG images already extracted')

    DataLoader = DataLoader('./CityScapes/train/rgb/*.png', './CityScapes/train/seg/*.png',
                            './CityScapes/test/rgb/*.png', './CityScapes/test/seg/*.png')

    BUFFER_SIZE = 100
    BATCH_SIZE = 20
    AUTO = tf.data.experimental.AUTOTUNE  # automatic load
    # load train set
    train_dataset_path = tf.data.Dataset.from_tensor_slices((DataLoader.train_rgb_path, DataLoader.train_seg_path))
    train_dataset = train_dataset_path.map(DataLoader.load_trainset, num_parallel_calls=AUTO)
    train_dataset = train_dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
    # load test set
    test_dataset_path = tf.data.Dataset.from_tensor_slices((DataLoader.test_rgb_path, DataLoader.test_seg_path))
    test_dataset = test_dataset_path.map(DataLoader.load_testset, num_parallel_calls=AUTO)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    print('Dataset:\n', train_dataset)












