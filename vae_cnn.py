from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
from blocks import conv_block, deconv_block, dense_block
from batch import get_batch
import random
import glob
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 8
NUM_TEST = 1000
CONV_FINAL_DIM = 16 * 16 * 64
LATENT_DIM = 256
CKPT_FILE = "ckpt/weights"
NUM_EPOCHS = 50

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

def get_encoder(inp, keep_prob):
  h = inp
  with tf.variable_scope('conv1') as scope:
    h = conv_block(inp, stride=1, relu=True)
  with tf.variable_scope('conv1a') as scope:
    h = conv_block(h, stride=2, relu=True)
  with tf.variable_scope('conv2') as scope:
    h = conv_block(h, stride=1, relu=True)
  with tf.variable_scope('conv2a') as scope:
    h = conv_block(h, stride=2, relu=True)
  with tf.variable_scope('conv3') as scope:
    h = conv_block(h, stride=1, relu=True)
  with tf.variable_scope('conv3a') as scope:
    h = conv_block(h, stride=2, relu=True)
  with tf.variable_scope('conv4') as scope:
    h = conv_block(h, stride=1, relu=True)
  with tf.variable_scope('conv4a') as scope:
    h = conv_block(h, stride=2, relu=True)
  with tf.variable_scope('conv5') as scope:
    h = conv_block(h, stride=1, relu=True)
  with tf.variable_scope('conv6') as scope:
    h = conv_block(h, stride=1, relu=True)

  h = tf.reshape(h, [BATCH_SIZE, CONV_FINAL_DIM])
  #h = tf.nn.dropout(h, keep_prob)

  with tf.variable_scope('enc_dense') as scope:
    h = dense_block(h, leaky_relu=True, output_size=1024)
  #h = tf.nn.dropout(h, keep_prob)

  return h

def get_decoder(inp, keep_prob):
  #h = tf.nn.dropout(inp, keep_prob)

  with tf.variable_scope('dec_dense') as scope:
    h = dense_block(inp, leaky_relu=True, output_size=CONV_FINAL_DIM)
  #h = tf.nn.dropout(h, keep_prob)

  h = tf.reshape(h, [-1, 16, 16, 64])

  with tf.variable_scope('deconv00') as scope:
    h = deconv_block(h, stride=1)
  with tf.variable_scope('deconv0') as scope:
    h = deconv_block(h, stride=1)
  with tf.variable_scope('deconv1a') as scope:
    h = deconv_block(h, stride=2)
  with tf.variable_scope('deconv1') as scope:
    h = deconv_block(h, stride=1)
  with tf.variable_scope('deconv2a') as scope:
    h = deconv_block(h, stride=2)
  with tf.variable_scope('deconv2') as scope:
    h = deconv_block(h, stride=1)
  with tf.variable_scope('deconv3a') as scope:
    h = deconv_block(h, stride=2)
  with tf.variable_scope('deconv3') as scope:
    h = deconv_block(h, stride=1)
  with tf.variable_scope('deconv4a') as scope:
    h = deconv_block(h, stride=2)
  with tf.variable_scope('deconv4') as scope:
    h = deconv_block(h, stride=1, output_channels=1)

  return h

class VAE:
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

        self.images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 256, 256, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        self._build_model(self.images, self.keep_prob)

    def _build_model(self, images, keep_prob):
        with tf.variable_scope('encoder') as scope:
            a = get_encoder(images, keep_prob)

        with tf.variable_scope('mu'):
            mu_encoder = dense_block(a, output_size=LATENT_DIM)

        with tf.variable_scope('logvar'):
            logvar_encoder = dense_block(a, output_size=LATENT_DIM)

        with tf.name_scope('z'):
            # Sample epsilon
            epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')
            # Sample latent variable
            std_encoder = tf.exp(0.5 * logvar_encoder)
            z = mu_encoder + tf.mul(std_encoder, epsilon)

        d = dense_block(z, output_size=1024)

        with tf.variable_scope('decoder') as scope:
            out = get_decoder(d, keep_prob)

        KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
        mean_KLD = tf.reduce_mean(KLD)
        mean_BCE = tf.reduce_mean(tf.squared_difference(out, images))

        loss = mean_BCE + 0.005 * mean_KLD
        tf.scalar_summary("KLD", mean_KLD)
        tf.scalar_summary("BCE", mean_BCE)
        tf.scalar_summary("lowerbound", loss)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        self.loss = loss
        self.image_hat = out
        self.train_step = train_step
        self.mu = mu_encoder
        self.logvar = logvar_encoder
        self.z = z

    def _restore(self, saver, initialize=False):
        if os.path.isfile(CKPT_FILE):
            print("Restoring saved parameters")
            saver.restore(self.sess, CKPT_FILE)
        elif initialize:
            print("Initializing parameters")
            self.sess.run(tf.initialize_all_variables())
        else:
            raise RuntimeError("No model trained.")

    def _initialize_train_and_test(self):
        file_list = glob.glob('/home/george/Documents/ddsm/pics/*/merged/*.png')
        file_list.sort()
        random.seed(1337)
        random.shuffle(file_list)

        print("Running on {0} images".format(len(file_list)))

        self.train_image, self.train_label = get_batch(file_list[:-NUM_TEST], BATCH_SIZE)
        self.test_image, self.test_label = get_batch(file_list[-NUM_TEST:], BATCH_SIZE)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        self.batches_per_epoch = (len(file_list) - NUM_TEST) // BATCH_SIZE

    def train(self):
        self._initialize_train_and_test()
        sess = self.sess
        train_image = self.train_image
        train_label = self.train_label

        # add op for merging summary
        summary_op = tf.merge_all_summaries()
        # add Saver ops
        saver = tf.train.Saver()
        summary_writer = tf.train.SummaryWriter('tb_logs', graph=sess.graph)
        self._restore(saver, initialize=True)

        for epoch in range(NUM_EPOCHS):
            print("Epoch {0}".format(epoch))
            for i in range(self.batches_per_epoch):
                global_step = epoch * self.batches_per_epoch + i
                img, _ = sess.run([train_image, train_label])
                _, cur_loss, summary_str = sess.run([self.train_step, self.loss, summary_op],
                        feed_dict={self.images: img, self.keep_prob: 0.5})
                summary_writer.add_summary(summary_str, global_step)

                if global_step % 10 == 0:
                    print("Step {0} | Loss: {1}".format(global_step, cur_loss))
                if global_step % 2000 == 0:
                    self.reconstruct(global_step)
            saver.save(sess, CKPT_FILE)

        self._close()

    # Parameters must be initialized
    def reconstruct(self, step):
        test_image, test_label = self.sess.run([self.test_image, self.test_label])
        sess = self.sess
        images = sess.run(self.image_hat, feed_dict={self.images: test_image, self.keep_prob: 1.0})
        _, a = plt.subplots(2, 6)
        for i in range(6):
            a[0][i].imshow(test_image[i].reshape((256, 256)), cmap='gray', vmin=0, vmax=1)
            a[1][i].imshow(images[i].reshape((256, 256)), cmap='gray', vmin=0, vmax=1)
        plt.savefig("img/reconstruct{0}.png".format(step))

    # Parameters must not be initialized
    def test_dump(self):
        self._initialize_train_and_test()
        sess = self.sess
        self._restore(tf.train.Saver())

        print('Dumping test data and labels.')
        test_samples = np.zeros((NUM_TEST, LATENT_DIM*2))
        test_labels = np.zeros(NUM_TEST)
        for i in range(NUM_TEST // BATCH_SIZE):
            img, labels = sess.run([self.test_image, self.test_label])
            mu, var = sess.run((self.mu, self.logvar), feed_dict={self.images: img, self.keep_prob: 1.0})
            samples = np.hstack((mu, var))
            test_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :] = samples
            test_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = labels
        print(test_samples.shape)
        np.savez('ddsm_test', data=test_samples, labels=test_labels)

        print('Dumping training data and labels.')
        train_samples = np.zeros((self.batches_per_epoch * BATCH_SIZE, LATENT_DIM * 2))
        train_labels = np.zeros(self.batches_per_epoch * BATCH_SIZE)
        for i in range(self.batches_per_epoch):
            img, labels = sess.run([self.train_image, self.train_label])
            mu, var = sess.run((self.mu, self.logvar), feed_dict={self.images: img, self.keep_prob: 1.0})
            samples = np.hstack((mu, var))
            train_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :] = samples
            train_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = labels

        print(train_samples.shape)
        np.savez('ddsm_train', data=train_samples, labels=train_labels)
        self._close()

    def _close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

v = VAE()
v.train()
