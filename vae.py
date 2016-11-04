from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST')

class VAE:
    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)

    def __init__(self):
        # Architecture variables
        input_dim = 784
        hidden_encoder_dim = 400
        hidden_decoder_dim = 400
        latent_dim = 20
        self.latent_dim = latent_dim
        lam = 0

        # Other constants
        self.ckpt_file = "temp.ckpt"
        self.display_samples = 8

        x = tf.placeholder("float", shape=[None, input_dim])
        l2_loss = tf.constant(0.0)

        # Hidden layer encoder
        W_encoder_input_hidden = VAE._weight_variable([input_dim,hidden_encoder_dim])
        b_encoder_input_hidden = VAE._bias_variable([hidden_encoder_dim])
        hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)
        l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

        # Mu encoder
        W_encoder_hidden_mu = VAE._weight_variable([hidden_encoder_dim,latent_dim])
        b_encoder_hidden_mu = VAE._bias_variable([latent_dim])
        mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu
        l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

        # Sigma encoder
        W_encoder_hidden_logvar = VAE._weight_variable([hidden_encoder_dim,latent_dim])
        b_encoder_hidden_logvar = VAE._bias_variable([latent_dim])
        logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar
        l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

        # Sample latent variable
        std_encoder = tf.exp(0.5 * logvar_encoder)
        z = mu_encoder + tf.mul(std_encoder, epsilon)

        # Hidden layer decoder
        W_decoder_z_hidden = VAE._weight_variable([latent_dim,hidden_decoder_dim])
        b_decoder_z_hidden = VAE._bias_variable([hidden_decoder_dim])
        hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)
        l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

        W_decoder_hidden_reconstruction = VAE._weight_variable([hidden_decoder_dim, input_dim])
        b_decoder_hidden_reconstruction = VAE._bias_variable([input_dim])
        x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
        l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

        KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
        BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, x), reduction_indices=1)
        loss = tf.reduce_mean(BCE + KLD)
        regularized_loss = loss + lam * l2_loss

        tf.scalar_summary("lowerbound", loss)
        train_step = tf.train.AdamOptimizer(0.001).minimize(regularized_loss)

        self.x = x
        self.train_step = train_step
        self.loss = loss
        self.x_hat = x_hat
        self.z = z

    def _restore(self, sess, saver, initialize=False):
        if os.path.isfile(self.ckpt_file):
            print("Restoring saved parameters")
            saver.restore(sess, self.ckpt_file)
        elif initialize:
            print("Initializing parameters")
            sess.run(tf.initialize_all_variables())
        else:
            raise RuntimeError("No model trained.")

    def train(self):
        n_steps = int(5e4)
        batch_size = 100
        with tf.Session() as sess:
            # add op for merging summary
            summary_op = tf.merge_all_summaries()
            # add Saver ops
            saver = tf.train.Saver()

            summary_writer = tf.train.SummaryWriter('experiment',
                                                  graph=sess.graph)
            self._restore(sess, saver, initialize=True)

            for step in range(n_steps):
                batch = mnist.train.next_batch(batch_size)
                _, cur_loss, summary_str = sess.run(
                        [self.train_step, self.loss, summary_op],
                        feed_dict={self.x: batch[0]})
                summary_writer.add_summary(summary_str, step)

                if step % 100 == 0:
                    save_path = saver.save(sess, self.ckpt_file)
                    print("Step {0} | Loss: {1}".format(step, cur_loss))

    def generate(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            self._restore(sess, saver)
            z_mu = np.random.normal(size=(self.display_samples,self.latent_dim))
            images = sess.run(self.x_hat, feed_dict={self.z: z_mu})
            _, a = plt.subplots(ncols=self.display_samples)
            for i in range(self.display_samples):
                a[i].imshow(images[i].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
            plt.show()

    def reconstruct(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            self._restore(sess, saver)
            images = sess.run(self.x_hat, feed_dict={self.x: mnist.test.images[:self.display_samples]})
            _, a = plt.subplots(2, self.display_samples)
            for i in range(self.display_samples):
                a[0][i].imshow(mnist.test.images[i].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
                a[1][i].imshow(images[i].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
            plt.show()

v = VAE()
v.train()
v.generate()
v.reconstruct()
