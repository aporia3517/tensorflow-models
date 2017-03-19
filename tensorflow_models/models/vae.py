# MIT License
#
# Copyright (c) 2017, Stefan Webb. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_models.initializations import xavier_init
from models import Model

# TODO: Break into tf_models package
# TODO: How to give uniform interface to initializers?
def _weights(shape, name='weights', initializer=xavier_init):
	return tf.get_variable(name, initializer=initializer(*shape))

def _biases(shape, name='biases', constant=0.1):
	return tf.get_variable(name, initializer=tf.constant(constant, shape=[shape]))

	# TODO: more than one variable into ReLU layer
def _relu_layer(x, weights, biases):
	return tf.nn.relu(tf.add(tf.matmul(x, weights), biases))

# TODO: VAE takes settings to get batchsize and size of input
class Vae(Model):
	def __init__(self, inputs, settings):
		# Dimensions of random variables
		# TODO: Get variables from settings
		# TODO: Take an architecture variable
		self.batch_size = 100
		self.n_x = 784
		self.n_z = 100

		# Save input node
		self.inputs = inputs

		# Architecture parameters
		self.n_hidden_recog_1 = 256
		self.n_hidden_recog_2 = 256
		self.n_hidden_gener_1 = 256
		self.n_hidden_gener_2 = 256

		self.x_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_x))
		self.z_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_z))

		# Call function that creates model and network
		self._lg_probs()

	# Encoder: q(z | x)
	# Returns the parameters for the normal distribution on z given x
	def _enc_z_given_x(self, inputs):
		with tf.variable_scope('q_z_given_x'):
			with tf.variable_scope('layer1'):
				weights = _weights(shape=(self.n_x, self.n_hidden_recog_1))
				biases = _biases(shape=self.n_hidden_recog_1)
				layer1 = _relu_layer(self.inputs, weights, biases)

			with tf.variable_scope('layer2'):
				weights = _weights(shape=(self.n_hidden_recog_1, self.n_hidden_recog_2))
				biases = _biases(shape=self.n_hidden_recog_2)
				layer2 = _relu_layer(layer1, weights, biases)

			with tf.variable_scope('mean'):
				weights = _weights(shape=(self.n_hidden_recog_2, self.n_z))
				biases = _biases(shape=self.n_z)
				mean_z = tf.add(tf.matmul(layer2, weights), biases)

			with tf.variable_scope('log_sigma_sq'):
				weights = _weights(shape=(self.n_hidden_recog_2, self.n_z))
				biases = _biases(shape=self.n_z)
				log_sigma_sq_z = tf.add(tf.matmul(layer2, weights), biases)
				diag_stdev_z = tf.sqrt(tf.exp(log_sigma_sq_z))
	
		return (mean_z, diag_stdev_z)

	# Decoder: p(x | z)
	# Returns parameters for bernoulli distribution on x given z
	def _dec_x_given_z(self, code):
		# Generate probabilistic decoder (decoder network), which
		# maps points in latent space onto a Bernoulli distribution in data space.
		# The transformation is parametrized and can be learned.
		with tf.variable_scope('p_x_given_z'):
			with tf.variable_scope('layer1'):
				weights = _weights(shape=(self.n_z, self.n_hidden_gener_1))
				biases = _biases(shape=self.n_hidden_gener_1)
				layer1 = _relu_layer(code, weights, biases)

			with tf.variable_scope('layer2'):
				weights = _weights(shape=(self.n_hidden_gener_1, self.n_hidden_gener_2))
				biases = _biases(shape=self.n_hidden_gener_2)
				layer2 = _relu_layer(layer1, weights, biases)

			with tf.variable_scope('mean'):
				weights = _weights(shape=(self.n_hidden_gener_2, self.n_x))
				biases = _biases(shape=self.n_x)
				logits = tf.add(tf.matmul(layer2, weights), biases)

		return logits

	def _lg_probs(self):
		# Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
		with tf.variable_scope('encoder'):
			mean_z, diag_stdev_z = self._enc_z_given_x(self.inputs)
			tf.get_variable_scope().reuse_variables()
			mean_z_placeholder, diag_stdev_z_placeholder = self._enc_z_given_x(self.x_placeholder)

		dist_z_given_x = tf.contrib.distributions.MultivariateNormalDiag(mean_z, diag_stdev_z)
		dist_z_given_x_placeholder = tf.contrib.distributions.MultivariateNormalDiag(mean_z_placeholder, diag_stdev_z_placeholder)

		self.encoder = dist_z_given_x_placeholder.sample()

		# Draw one sample z from Gaussian distribution
		eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
		self.z_sample = tf.add(mean_z, tf.multiply(diag_stdev_z, eps))

		# Define the prior distribution and a sample from it
		dist_z = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros([self.batch_size, self.n_z]), tf.ones([self.batch_size, self.n_z]))
		self.z_prior = dist_z.sample()

		# Use generator to determine mean of Bernoulli distribution of reconstructed input
		with tf.variable_scope('decoder'):
			logits_x = self._dec_x_given_z(self.z_sample)
			tf.get_variable_scope().reuse_variables()
			logits_x_placeholder = self._dec_x_given_z(self.z_placeholder)

		dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x)
		dist_x_given_z_placeholder = tf.contrib.distributions.Bernoulli(logits=logits_x_placeholder)

		self.decoder = dist_x_given_z_placeholder.sample()

		self.lg_p_x_given_z = dist_x_given_z.log_prob(self.inputs)
		self.lg_p_z = dist_z.log_prob(self.z_sample)
		self.lg_q_z_given_x = dist_z_given_x.log_prob(self.z_sample)

	def inference(self):
		return self.lg_p_x_given_z, self.lg_p_z, self.lg_q_z_given_x

	def sample_prior(self):
		return np.random.normal(size=(self.batch_size, self.n_z))