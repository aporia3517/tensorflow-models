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
from tensorflow_models import Model

# TODO: VAE takes settings to get batchsize and size of input
class Model(Model):
	def __init__(self, inputs, settings):
		# Dimensions of random variables
		# TODO: Get variables from settings
		# TODO: Take an architecture variable
		self.batch_size = settings['batch_size']
		self.n_x = 784
		self.n_z = 100

		# Save input node
		self.inputs = inputs

		# Architecture parameters
		self.generator_sizes = [128, self.n_x]
		self.discriminator_sizes = [128, 1]

		self.x_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_x))
		self.z_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_z))

		# Call function that creates model and network
		self._lg_probs()

	def _generator_network(self, code):
		return tf_models.layers.mlp(code, self.generator_sizes, final_activation_fn=tf.nn.sigmoid)

	def _discriminator_network(self, inputs):
		return tf_models.layers.mlp(inputs, self.discriminator_sizes, final_activation_fn=tf.nn.sigmoid)

	def _lg_probs(self):
		eps = tf.random_uniform((self.batch_size, self.n_z), minval=-1., maxval=1., dtype=tf.float32)

		with tf.variable_scope('generator'):
			fake = self._generator_network(eps)
			tf.get_variable_scope().reuse_variables()
			self.decoder = self._generator_network(self.z_placeholder)

		with tf.variable_scope('discriminator'):
			p_data = self._discriminator_network(self.inputs)
			tf.get_variable_scope().reuse_variables()
			p_fake = self._discriminator_network(fake)

		self.ll_data = tf.reduce_sum(tf.log(p_data), 1)
		self.ll_fake = tf.reduce_sum(tf.log(p_fake), 1)
		self.ll_one_minus_fake = tf.reduce_sum(tf.log(1. - p_fake), 1)

	def inference(self):
		return {'ll_data': self.ll_data, 'll_fake': self.ll_fake, 'll_one_minus_fake': self.ll_one_minus_fake}

	def sample_prior(self):
		return np.random.uniform(low=-1., high=1., size=[self.batch_size, self.n_z])