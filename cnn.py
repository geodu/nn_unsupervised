from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import tensorflow as tf

NUM_CLASSES = 3
BATCH_SIZE = 50

def files_and_labels():
  files = glob.glob('/home/george/Documents/ddsm/pics/*/scaled/*.png')
  length = int(len(files) / 4)
  files.sort()
  cc_left = files[::4]
  cc_right = files[2::4]
  mlo_left = files[1::4]
  mlo_right = files[3::4]
  left_image = cc_left + mlo_left
  right_image = cc_right + mlo_right
  label = map(lambda x: 2 if 'benign' in x else 1 if 'cancer' in x else 0, left_image)
  return left_image, right_image, list(label)

def _process_image_string(istr):
  value = tf.read_file(istr)
  img = tf.image.decode_png(value)
  w_img = tf.image.per_image_whitening(img)
  w_img.set_shape((512, 256, 1))
  return w_img

def get_data():
  l_image_list, r_image_list, label_list = files_and_labels()
  l_images = tf.convert_to_tensor(l_image_list, dtype=tf.string)
  r_images = tf.convert_to_tensor(r_image_list, dtype=tf.string)
  labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

  input_queue = tf.train.slice_input_producer([l_images, r_images, labels])

  l_img = _process_image_string(input_queue[0])
  r_img = _process_image_string(input_queue[1])
  return l_img, r_img, input_queue[2]

def get_batch():
  l_image, r_image, label = get_data()
  batch_size = BATCH_SIZE
  min_after_dequeue = 1000
  capacity = min_after_dequeue + 3 * batch_size
  l_image_batch, r_image_batch, label_batch = tf.train.shuffle_batch(
      [l_image, r_image, label], batch_size=batch_size, capacity=capacity,
       min_after_dequeue=min_after_dequeue)
  return l_image_batch, r_image_batch, label_batch

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
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(inp, kernel, [1, 1, 1, 1], padding='SAME')
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
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
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
    reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
    dim = reshape.get_shape()[1].value
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
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  return cross_entropy_mean

def inference():
  images1, images2, labels = get_batch()
  with tf.variable_scope('cnns') as scope:
    cnn1 = get_cnn(images1)
    scope.reuse_variables()
    cnn2 = get_cnn(images2)
  together = tf.concat(1, [cnn1, cnn2])
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192*2, NUM_CLASSES],
                                          stddev=1/(2*192.0), wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(together, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  loss = get_loss(softmax_linear, labels)
  train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
  return train_step, loss

with tf.Session() as sess:
  train_step, loss = inference()
  sess.run(tf.initialize_all_variables())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  for i in range(10000):
    _, loss = sess.run([train_step, loss])
    print(loss)
  coord.request_stop()
  coord.join(threads)
