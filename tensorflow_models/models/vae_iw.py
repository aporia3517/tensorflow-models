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

import tensorflow_models as tf_models

def create_placeholders(settings):
	x = tf.placeholder(tf.float32, shape=tf_models.batchshape(settings), name='samples')
	z = tf.placeholder(tf.float32, shape=tf_models.latentshape(settings), name='codes')
	return x, z

def create_prior(settings):
	dist_prior = tf_models.standard_normal(tf_models.latentshape(settings))
	return tf.identity(dist_prior.sample(), name='p_z/sample')

def create_encoder(settings, reuse=True):
	encoder_network = settings['architecture']['encoder']['fn']

	x_placeholder = tf_models.samples_placeholder()
	assert(not x_placeholder is None)

	with tf.variable_scope('encoder', reuse=reuse):
		mean_z, diag_stdev_z = encoder_network(settings, x_placeholder, is_training=False)
		dist_z_given_x = tf.contrib.distributions.MultivariateNormalDiag(mean_z, diag_stdev_z)
		encoder = tf.identity(dist_z_given_x.sample(name='sample'), name='q_z_given_x/sample')
	return encoder

def create_decoder(settings, reuse=True):
	decoder_network = settings['architecture']['decoder']['fn']

	z_placeholder = tf_models.codes_placeholder()
	assert(not z_placeholder is None)

	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_placeholder, is_training=False)
		#dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x, dtype=tf.float32)
		#decoder = tf.identity(dist_x_given_z.sample(), name='p_x_given_z/sample')
	#return decoder
	return tf.identity(tf.nn.sigmoid(logits_x), name='p_x_given_z/sample')

def create_probs(settings, inputs, is_training, reuse=False):
	encoder_network = settings['architecture']['encoder']['fn']
	decoder_network = settings['architecture']['decoder']['fn']
	
	dist_prior = tf_models.standard_normal(tf_models.latentshape(settings))
	dist_prior_iw = tf.contrib.distributions.Normal(loc=0., scale=1.)

	# Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
	with tf.variable_scope('encoder', reuse=reuse):
		mean_z, diag_stdev_z = encoder_network(settings, inputs, is_training=is_training)
	#dist_z_given_x = tf.contrib.distributions.MultivariateNormalDiag(mean_z, diag_stdev_z)
	dist_z_given_x = tf.contrib.distributions.Normal(loc=mean_z, scale=diag_stdev_z)

	# Draw one sample z from Gaussian distribution
	eps = tf.random_normal(tf_models.latentshape(settings), 0, 1, dtype=tf.float32)
	z_sample = tf.add(mean_z, tf.multiply(diag_stdev_z, eps))

	eps = tf.random_normal((settings['iw_size'], settings['batch_size'], settings['latent_dimension']), 0, 1, dtype=tf.float32)
	z_sample_iw = tf.add(tf.reshape(mean_z, (1, settings['batch_size'], settings['latent_dimension'])), tf.multiply(tf.reshape(diag_stdev_z, (1, settings['batch_size'], settings['latent_dimension'])), eps))
	#z_sample_iw = dist_z_given_x.sample([iw_size])
	#print('z_sample_iw.shape', z_sample_iw.shape)
	#raise Exception()

	# Use generator to determine mean of Bernoulli distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_sample, is_training=is_training)
		tf.get_variable_scope().reuse_variables()
		logits_x_iw = decoder_network(settings, tf.reshape(z_sample_iw, (-1, settings['latent_dimension'])), is_training=is_training)

	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	dist_x_given_z_iw = tf.contrib.distributions.Bernoulli(logits=tf.reshape(tf_models.flatten(logits_x_iw), (settings['iw_size'], settings['batch_size'], -1)), dtype=tf.float32)
	
	# NOTE: x | z is defined as over each pixel separate, where prior on z is a multivariate
	# Hence the need to do the tf.reduce_sum op on the former to get down to a single number for each sample
	lg_p_x_given_z = tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(inputs)), 1, name='p_x_given_z/log_prob')
	lg_p_z = tf.identity(dist_prior.log_prob(z_sample), name='p_z/log_prob')
	lg_q_z_given_x = tf.reduce_sum(dist_z_given_x.log_prob(z_sample), 1, name='q_z_given_x/log_prob')

	inputs_iw = tf.tile(tf.expand_dims(tf_models.flatten(inputs), axis=0), multiples=[settings['iw_size'],1,1])

	lg_p_x_given_z_iw = tf.reduce_sum(dist_x_given_z_iw.log_prob(inputs_iw), 2, name='p_x_given_z_iw/log_prob')
	lg_p_z_iw = tf.reduce_sum(dist_prior_iw.log_prob(z_sample_iw), 2, name='p_z_iw/log_prob')
	lg_q_z_given_x_iw = tf.reduce_sum(dist_z_given_x.log_prob(z_sample_iw), 2, name='q_z_given_x_iw/log_prob')

	return lg_p_x_given_z, lg_p_z, lg_q_z_given_x, lg_p_x_given_z_iw, lg_p_z_iw, lg_q_z_given_x_iw

def lg_likelihood(x, z, settings, reuse=True, is_training=False):
	decoder_network = settings['architecture']['decoder']['fn']

	with tf.variable_scope('model'):
		with tf.variable_scope('decoder', reuse=reuse):
			logits_x = decoder_network(settings, z, is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	return tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(x)), 1)

def lg_prior(z, settings, reuse=True, is_training=False):
	dist_prior = tf_models.standard_normal(z.shape)
	return dist_prior.log_prob(z)

def sample_prior(settings):
	dist_prior = tf_models.standard_normal(tf_models.latentshape(settings))
	return dist_prior.sample()

