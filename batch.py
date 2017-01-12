from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def _files_and_labels(file_list):
  files = list(file_list)
  labels = map(lambda x: 2 if 'benign' in x else 1 if 'cancer' in x else 0, files)
  return files, list(labels)

def _process_image_string(istr):
  value = tf.read_file(istr)
  img = tf.image.decode_png(value, dtype=tf.uint16)
  #img = tf.image.per_image_whitening(img)
  img = tf.cast(img, tf.float32)
  img = img / 65536
  img.set_shape((256, 256, 1))
  return img

def _get_data(file_list):
  image_list, label_list = _files_and_labels(file_list)
  images = tf.convert_to_tensor(image_list, dtype=tf.string)
  labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

  input_queue = tf.train.slice_input_producer([images, labels])

  img = _process_image_string(input_queue[0])
  return img, input_queue[1]

def get_batch(file_list, batch_size):
  image, label = _get_data(file_list)
  min_after_dequeue = 1000
  capacity = min_after_dequeue + 3 * batch_size
  image_batch, label_batch = tf.train.shuffle_batch(
      [image, label], batch_size=batch_size, capacity=capacity,
       min_after_dequeue=min_after_dequeue)
  return image_batch, label_batch


