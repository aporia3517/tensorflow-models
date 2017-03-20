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
import tensorflow.contrib.slim as slim

import tensorflow_models as tf_models
from tensorflow_models.initializations import xavier_init, xavier_std
from tensorflow_models import Model

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
class Model(Model):
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
		self.encoder_sizes = [256, 256, self.n_z]
		self.decoder_sizes = [256, 256, self.n_x]

		self.x_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_x))
		self.z_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_z))

		# Call function that creates model and network
		self._lg_probs()

	# Encoder: q(z | x)
	# Returns the parameters for the normal distribution on z given x
	def _enc_z_given_x(self, inputs):
		#with tf.variable_scope('q_z_given_x'):
		return tf_models.layers.gaussian_parameters_mlp(inputs, self.encoder_sizes)

	# Decoder: p(x | z)
	# Returns parameters for bernoulli distribution on x given z
	def _dec_x_given_z(self, code):
		#with tf.variable_scope('p_x_given_z'):
		return tf_models.layers.bernoulli_parameters_mlp(code, self.decoder_sizes)
		
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

		# NOTE: x | z is defined as over each pixel separate, where prior on z is a multivariate
		# Hence the need to do the tf.reduce_sum op on the former to get down to a single number for each sample
		self.lg_p_x_given_z = tf.reduce_sum(dist_x_given_z.log_prob(self.inputs), 1)
		self.lg_p_z = dist_z.log_prob(self.z_sample)
		self.lg_q_z_given_x = dist_z_given_x.log_prob(self.z_sample)

	def inference(self):
		return {'ll_decoder': self.lg_p_x_given_z, 'll_prior': self.lg_p_z, 'll_encoder': self.lg_q_z_given_x}

	def sample_prior(self):
		return np.random.normal(size=(self.batch_size, self.n_z))