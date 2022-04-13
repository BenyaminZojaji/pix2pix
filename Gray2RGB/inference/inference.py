import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--input", default='input', type = str, help='input PATH')
parser.add_argument("--output", default='output', type = str, help='output PATH')
parser.add_argument("--model", default='gray2rgb-2.h5', type = str, help='model PATH')
args = parser.parse_args()

model = load_model(args.model)

for index, name in enumerate(os.listdir(args.input)):
    image = tf.io.read_file(f'{args.input}/{name}')
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.expand_dims(image, axis=0)
    pred = model(image, training=False)
    pred = pred * 0.5+0.5
    pred = tf.image.convert_image_dtype(pred, tf.uint8)
    pred = tf.squeeze(pred, axis=0)
    pred = np.array(pred)
    cv2.imwrite(f'{args.output}/{index}.jpg', pred)