from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST')

input_dim = 784
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0

ckpt_file = "temp.ckpt"
n_steps = int(1e6)
batch_size = 100

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

class VAE:
    def __init__(self):
        x = tf.placeholder("float", shape=[None, input_dim])
        l2_loss = tf.constant(0.0)

        # Hidden layer encoder
        W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
        b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
        hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)
        l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

        # Mu encoder
        W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
        b_encoder_hidden_mu = bias_variable([latent_dim])
        mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu
        l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

        # Sigma encoder
        W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
        b_encoder_hidden_logvar = bias_variable([latent_dim])
        logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar
        l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

        # Sample latent variable
        std_encoder = tf.exp(0.5 * logvar_encoder)
        z = mu_encoder + tf.mul(std_encoder, epsilon)

        # Hidden layer decoder
        W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
        b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
        hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)
        l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

        W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
        b_decoder_hidden_reconstruction = bias_variable([input_dim])
        x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
        l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

        KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
        BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, x), reduction_indices=1)
        loss = tf.reduce_mean(BCE + KLD)
        regularized_loss = loss + lam * l2_loss

        tf.scalar_summary("lowerbound", loss)
        train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

        self.x = x
        self.train_step = train_step
        self.loss = loss

def train(vae):
    with tf.Session() as sess:
        # add op for merging summary
        summary_op = tf.merge_all_summaries()
        # add Saver ops
        saver = tf.train.Saver()

        summary_writer = tf.train.SummaryWriter('experiment',
                                              graph=sess.graph)
        if os.path.isfile(ckpt_file):
            print("Restoring saved parameters")
            saver.restore(sess, ckpt_file)
        else:
            print("Initializing parameters")
            sess.run(tf.initialize_all_variables())

        for step in range(n_steps):
            batch = mnist.train.next_batch(batch_size)
            _, cur_loss, summary_str = sess.run(
                    [vae.train_step, vae.loss, summary_op],
                    feed_dict={vae.x: batch[0]})
            summary_writer.add_summary(summary_str, step)

            if step % 100 == 0:
                save_path = saver.save(sess, ckpt_file)
                print("Step {0} | Loss: {1}".format(step, cur_loss))

v = VAE()
train(v)
