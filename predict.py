import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from model import UNET
from visualize import color_map, get_palette
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


# Load model and weights
model = UNET()
weights = tf.train.latest_checkpoint('./models')
model.load_weights(weights).expect_partial()


# Obtain input image
img = tf.io.read_file('./CityScapes/test/rgb/17_rgb.png')
img = tf.image.decode_png(img, channels=3)
img = tf.image.resize(img, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
img = tf.cast(img, tf.float32) / 127.5 - 1
img = tf.expand_dims(img, axis=0)   # This is a single image. To make it the same dim with dataset, add a dim.


# Predict
output = model.predict(img)


# Decode output from 'One Hot' to image
output = tf.argmax(output, axis=-1)
print('output shape: ', tf.shape(output))


# show
output = output.numpy().astype(np.uint8)
output = Image.fromarray(np.squeeze(output))
output.save('./img/output.png')


# generate palette img
src = Image.open('./img/output.png')
mat = np.array(src)
mat = mat.astype(np.uint8)
dst = Image.fromarray(mat, 'P')
palette = get_palette(color_map)
dst.putpalette(palette)
dst.save('./img/palette.png')
plt.imshow(dst)
plt.show()



