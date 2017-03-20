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
from tensorflow_models import Model

# TODO: Best initialization for ReLU biases? -1 or 0.1?

# TODO: Break into tf_models package
# TODO: How to give uniform interface to initializers?
def _weights(shape, name='weights', initializer=xavier_init):
	return tf.get_variable(name, initializer=initializer(*shape))

def _biases(shape, name='biases', constant=0.1):
	return tf.get_variable(name, initializer=tf.constant(constant, shape=[shape]))

	# TODO: more than one variable into ReLU layer
def _relu_layer(x, weights, biases):
	return tf.nn.relu(tf.add(tf.matmul(x, weights), biases))

class Model(Model):
	def __init__(self, inputs, settings):
		# Dimensions of random variables
		self.batch_size = 100
		self.n_x = 784
		self.n_z = 100
		self.n_eps = 100

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

	# Black-box encoder: q(z | x, eps)
	# Returns a sample from z given x and epsilon
	def _enc_z_given_x_eps(self, inputs, eps):
		#with tf.variable_scope('encoder'):
		with tf.variable_scope('q_z_given_x_eps'):
			with tf.variable_scope('layer1'):
				weights_x = _weights(shape=(self.n_x, self.n_hidden_recog_1), name='weights_x')
				weights_eps = _weights(shape=(self.n_eps, self.n_hidden_recog_1), name='weights_eps')
				biases = _biases(shape=self.n_hidden_recog_1)
				layer1 = tf.nn.relu(tf.add(tf.add(tf.matmul(inputs, weights_x), tf.matmul(eps, weights_eps)), biases))

			with tf.variable_scope('layer2'):
				weights = _weights(shape=(self.n_hidden_recog_1, self.n_hidden_recog_2))
				biases = _biases(shape=self.n_hidden_recog_2)
				layer2 = _relu_layer(layer1, weights, biases)

			with tf.variable_scope('z'):
				weights = _weights(shape=(self.n_hidden_recog_2, self.n_z))
				biases = _biases(shape=self.n_z)
				z = tf.add(tf.matmul(layer2, weights), biases)
	
		return z

	# Decoder: p(x | z)
	# Returns parameters for bernoulli distribution on x given z
	def _dec_x_given_z(self, code):
		# Generate probabilistic decoder (decoder network), which
		# maps points in latent space onto a Bernoulli distribution in data space.
		# The transformation is parametrized and can be learned.
		#with tf.variable_scope('decoder'):
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

	# Discriminator used for adversarial training in logits
	def _discriminator(self, x, z):
		#with tf.variable_scope('discriminator'):
		with tf.variable_scope('x_network'):
			with tf.variable_scope('layer1'):
				weights = _weights(shape=(self.n_x, self.n_hidden_gener_1))
				biases = _biases(shape=self.n_hidden_gener_1)
				layer1 = _relu_layer(x, weights, biases)
					
			with tf.variable_scope('layer2'):
				weights = _weights(shape=(self.n_hidden_gener_1, self.n_hidden_gener_2))
				biases = _biases(shape=self.n_hidden_gener_2)
				x_layer2 = _relu_layer(layer1, weights, biases)

		with tf.variable_scope('z_network'):
			with tf.variable_scope('layer1'):
				weights = _weights(shape=(self.n_z, self.n_hidden_gener_1))
				biases = _biases(shape=self.n_hidden_gener_1)
				layer1 = _relu_layer(z, weights, biases)

			with tf.variable_scope('layer2'):
				weights = _weights(shape=(self.n_hidden_gener_1, self.n_hidden_gener_2))
				biases = _biases(shape=self.n_hidden_gener_2)
				z_layer2 = _relu_layer(layer1, weights, biases)

		with tf.variable_scope('combine_network'):
			with tf.variable_scope('layer1'):
				weights_x = _weights(shape=(self.n_hidden_gener_2, self.n_hidden_gener_1), name='weights_x')
				weights_z = _weights(shape=(self.n_hidden_gener_2, self.n_hidden_gener_1), name='weights_z')
				biases = _biases(shape=self.n_hidden_gener_1)
				layer1 = tf.nn.relu(tf.add(tf.add(tf.matmul(x_layer2, weights_x), tf.matmul(z_layer2, weights_z)), biases))

			with tf.variable_scope('layer2'):
				weights = _weights(shape=(self.n_hidden_gener_1, self.n_hidden_gener_2))
				biases = _biases(shape=self.n_hidden_gener_2)
				layer2 = _relu_layer(layer1, weights, biases)
					
			with tf.variable_scope('output'):
				weights = _weights(shape=(self.n_hidden_gener_2, 1))
				biases = _biases(shape=1)
				logits = tf.add(tf.matmul(layer2, weights), biases)

		return logits

	# Create operations to calculate terms needed for loss function
	def _lg_probs(self):
		# The noise is distributed i.i.d. N(0, 1)
		self.noise = tf.random_normal((self.batch_size, self.n_eps), 0, 1, dtype=tf.float32)

		# Use black-box inference network to sample z, given inputs and noise
		with tf.variable_scope('encoder'):
			self.z_sample = self._enc_z_given_x_eps(self.inputs, self.noise)
			tf.get_variable_scope().reuse_variables()
			self.encoder = self._enc_z_given_x_eps(self.x_placeholder, self.noise)

		# The prior on z is also i.i.d. N(0, 1)
		#dist_z = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros([self.batch_size, self.n_z]), tf.ones([self.batch_size, self.n_z]))
		self.z_prior = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float32)

		# Use generator to determine distribution of reconstructed input
		with tf.variable_scope('decoder'):
			logits_x = self._dec_x_given_z(self.z_sample)
			tf.get_variable_scope().reuse_variables()
			logits_x_placeholder = self._dec_x_given_z(self.z_placeholder)

		dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x)

		dist_x_given_z_placeholder = tf.contrib.distributions.Bernoulli(logits=logits_x_placeholder)
		self.decoder = dist_x_given_z_placeholder.sample()

		# Log likelihood of reconstructed inputs
		self.lg_p_x_given_z = tf.reduce_sum(dist_x_given_z.log_prob(self.inputs), 1)

		# Discriminator T(x, z)
		with tf.variable_scope('discriminator'):
			self.adversary = self._discriminator(self.inputs, self.z_sample)
			tf.get_variable_scope().reuse_variables()
			self.prior_adversary = self._discriminator(self.inputs, self.z_prior)

	def inference(self):
		return {'ll_decoder': self.lg_p_x_given_z, 'adversary': self.adversary, 'prior_adversary': self.prior_adversary}

	def sample_prior(self):
		return np.random.normal(size=(self.batch_size, self.n_z))