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

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
import tensorflow_models as tf_models
import tensorflow.contrib.slim as slim

weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

# Define a multilayer perceptron, or dense feedforward net
def mlp(inputs, sizes, scope='layer', activation_fn=tf.nn.relu, final_activation_fn=None, normalizer_fn=None):
	if final_activation_fn is None:
		#return slim.stack(inputs, slim.fully_connected, sizes, scope=scope, activation_fn=activation_fn, weights_initializer=weights_initializer)
		return slim.stack(inputs, slim.fully_connected, sizes, scope=scope, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
	else:
		# TODO: Check that variables are being created properly and not reused
		#layer = slim.stack(inputs, slim.fully_connected, sizes[:-1], scope=scope, activation_fn=activation_fn, weights_initializer=weights_initializer)
		#return slim.fully_connected(layer, sizes[-1], scope=scope, activation_fn=final_activation_fn, weights_initializer=weights_initializer)
		layer = slim.stack(inputs, slim.fully_connected, sizes[:-1], scope=scope, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
		return slim.fully_connected(layer, sizes[-1], scope=scope, activation_fn=final_activation_fn)

# Define a network from inputs to mean and standard deviation parameters for a diagonal Gaussian distribution
# NOTE: This is the basic architecture for the encoder network in a VAE
def gaussian_parameters_mlp(inputs, sizes, activation_fn=tf.nn.relu, normalizer_fn=None):
	layer = mlp(inputs, sizes[:-1], activation_fn=activation_fn, normalizer_fn=normalizer_fn)
	#mean_z = slim.fully_connected(layer, sizes[-1], activation_fn=tf.identity, scope='mean', weights_initializer=weights_initializer)
	#log_sigma_sq_z = slim.fully_connected(layer, sizes[-1], activation_fn=tf.identity, scope='log_sigma_sq', weights_initializer=weights_initializer)
	mean_z = slim.fully_connected(layer, sizes[-1], activation_fn=tf.identity, scope='mean')
	log_sigma_sq_z = slim.fully_connected(layer, sizes[-1], activation_fn=tf.identity, scope='log_sigma_sq')
	diag_stdev_z = tf.sqrt(tf.exp(log_sigma_sq_z))
	return mean_z, diag_stdev_z

def beta_parameters_mlp(inputs, sizes, activation_fn=tf.nn.relu, normalizer_fn=None):
	layer = mlp(inputs, sizes[:-1], activation_fn=activation_fn, normalizer_fn=normalizer_fn)
	#mean_z = slim.fully_connected(layer, sizes[-1], activation_fn=tf.identity, scope='mean', weights_initializer=weights_initializer)
	#log_sigma_sq_z = slim.fully_connected(layer, sizes[-1], activation_fn=tf.identity, scope='log_sigma_sq', weights_initializer=weights_initializer)
	logits_alpha = slim.fully_connected(layer, sizes[-1], activation_fn=tf.identity, scope='logits_alpha')
	logits_beta = slim.fully_connected(layer, sizes[-1], activation_fn=tf.identity, scope='logits_beta')
	return logits_alpha, logits_beta

def bernoulli_parameters_mlp(inputs, sizes, activation_fn=tf.nn.relu, normalizer_fn=None):
	logits = mlp(inputs, sizes, activation_fn=activation_fn, final_activation_fn=tf.identity, normalizer_fn=normalizer_fn)
	return logits
