from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
import random
from batch import get_batch

NUM_CLASSES = 3
BATCH_SIZE = 32
TRAIN_RATIO = .9

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(name, shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def get_cnn(inp):
  batch_size = tf.shape(inp)[0]
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(inp, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = 16 * 32 * 64
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  return local4

def get_loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  return cross_entropy_mean

def inference(images1, images2, labels):
  labels = tf.cast(labels, tf.int64)
  with tf.variable_scope('cnns') as scope:
    cnn1 = get_cnn(images1)
    scope.reuse_variables()
    cnn2 = get_cnn(images2)
  together = tf.concat(1, [cnn1, cnn2])
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', shape=[192*2, NUM_CLASSES],
                                          stddev=1/(2*192.0), wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(together, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  correct_prediction = tf.equal(tf.argmax(softmax_linear, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  loss = get_loss(softmax_linear, labels)
  train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
  return train_step, loss, accuracy

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
  random.seed(1337)
  file_list = glob.glob('/home/george/Documents/ddsm/pics/*/scaled/*.png')
  file_list.sort()
  random.shuffle(file_list)
  num_training = int(len(file_list) * TRAIN_RATIO)
  num_test = len(file_list) - num_training
  print('Running on {0} training images, {1} test images'.format(num_training, num_test))
  train_image1, train_image2, train_label = get_batch(file_list[:num_training], BATCH_SIZE)
  test_image1, test_image2, test_label = get_batch(file_list[num_training:], BATCH_SIZE)

  print('Building model')
  images1 = tf.placeholder(tf.float32, shape=[None, 512, 256, 1])
  images2 = tf.placeholder(tf.float32, shape=[None, 512, 256, 1])
  labels = tf.placeholder(dtype=tf.int32, shape=[None])
  train_step, loss, accuracy = inference(images1, images2, labels)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  sess.run(tf.initialize_all_variables())

  batches_per_epoch = num_training // BATCH_SIZE
  for epoch in range(50):
    print('Epoch {0}'.format(epoch))
    for i in range(batches_per_epoch):
      global_step = epoch * batches_per_epoch + i
      a, b, c = sess.run([train_image1, train_image2, train_label])
      _, los = sess.run([train_step, loss], feed_dict={images1:a, images2:b, labels:c})
      if global_step % 10 == 0:
        print("Step {0} Loss {1}".format(global_step, los))

    losses = []
    accuracies = []
    for i in range(num_test // BATCH_SIZE):
      a, b, c = sess.run([test_image1, test_image2, test_label])
      los, acc = sess.run([loss, accuracy], feed_dict={images1:a, images2:b, labels:c})
      losses.append(los)
      accuracies.append(acc)
    los = np.mean(losses)
    acc = np.mean(accuracies)
    print("\nTest Loss {0} Test Accuracy {1}\n".format(los, acc))
  coord.request_stop()
  coord.join(threads)
