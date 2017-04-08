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

# Architecture stuff
#encoder_sizes = [256, 256] # self.n_z
#decoder_sizes = [256, 256] # self.n_x

#def create(settings):
#	# For the moment, we require a flattened input
#	assert('flatten' in settings['transformations'])

def create_placeholders(settings):
	x = tf.placeholder(tf.float32, shape=tf_models.batchshape(settings), name='samples')
	z = tf.placeholder(tf.float32, shape=latentshape(settings), name='codes')
	return x, z

def create_prior(settings):
	dist_prior = tf_models.standard_normal(latentshape(settings), name='p_z')
	return dist_prior.sample(name='sample')

def latentshape(settings):
	return [settings['batch_size'], settings['latent_dimension']]

# Encoder: q(z | x)
# Returns the parameters for the normal distribution on z given x
def encoder_network(settings, inputs):
	#with tf.variable_scope('q_z_given_x'):
	return tf_models.layers.gaussian_parameters_mlp(inputs, settings['encoder_sizes'] + [settings['latent_dimension']])

# Decoder: p(x | z)
# Returns parameters for bernoulli distribution on x given z
def decoder_network(settings, code):
	return tf_models.layers.bernoulli_parameters_mlp(code, settings['decoder_sizes'] + tf_models.flattened_shape(settings))

def create_encoder(settings, reuse=True):
	x_placeholder = tf_models.samples_placeholder()
	assert(not x_placeholder is None)

	with tf.variable_scope('encoder', reuse=reuse):
		mean_z, diag_stdev_z = encoder_network(settings, x_placeholder)
		dist_z_given_x = tf.contrib.distributions.MultivariateNormalDiag(mean_z, diag_stdev_z, name='q_z_given_x')
		encoder = dist_z_given_x.sample(name='sample')
	return encoder

def create_decoder(settings, reuse=True):
	z_placeholder = tf_models.codes_placeholder()
	assert(not z_placeholder is None)

	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_placeholder)
		dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x, name='p_x_given_z')
		decoder = dist_x_given_z.sample(name='sample')
	return decoder

def create_probs(settings, inputs, reuse=False):
	dist_prior = tf_models.standard_normal(latentshape(settings), name='p_z')

	# Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
	with tf.variable_scope('encoder', reuse=reuse):
		mean_z, diag_stdev_z = encoder_network(settings, inputs)
	dist_z_given_x = tf.contrib.distributions.MultivariateNormalDiag(mean_z, diag_stdev_z, name='q_z_given_x')

	# Draw one sample z from Gaussian distribution
	eps = tf.random_normal(latentshape(settings), 0, 1, dtype=tf.float32)
	z_sample = tf.add(mean_z, tf.multiply(diag_stdev_z, eps))

	# Use generator to determine mean of Bernoulli distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_sample)	
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x)
	
	# NOTE: x | z is defined as over each pixel separate, where prior on z is a multivariate
	# Hence the need to do the tf.reduce_sum op on the former to get down to a single number for each sample
	lg_p_x_given_z = tf.reduce_sum(dist_x_given_z.log_prob(inputs), 1, name='p_x_given_z/log_prob')
	lg_p_z = dist_prior.log_prob(z_sample)
	lg_q_z_given_x = dist_z_given_x.log_prob(z_sample)

	return lg_p_x_given_z, lg_p_z, lg_q_z_given_x