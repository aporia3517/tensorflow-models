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

import math

import tensorflow_models as tf_models
import tensorflow_models.relaxed_onehot_categorical_fixed_noise as dist_fixed

def create_placeholders(settings):
	K = settings['count_categories']
	latent_batchshape = (settings['batch_size'], settings['latent_dimension'], K)

	x = tf.placeholder(tf.float32, shape=tf_models.batchshape(settings), name='samples')
	z = tf.placeholder(tf.float32, shape=latent_batchshape, name='codes')
	return x, z

#def create_prior(settings):
#	dist_prior = tf.contrib.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
#	return tf.identity(dist_prior.sample(sample_shape=tf_models.latentshape(settings)) * 2. - 1., name='p_z/sample')

def create_prior(settings):
	temperature_prior = 0.5
	K = settings['count_categories']
	latent_batchshape = (settings['batch_size'], settings['latent_dimension'], K)
	
	dist_prior = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature_prior, logits=tf.constant(0., shape=latent_batchshape))
	logits_sample = tf.cast(dist_prior.sample(), dtype=tf.float32)
	z_sample = tf.exp(logits_sample) *2. - 1.

	return tf.identity(z_sample, name='p_z/sample')

def create_encoder(settings, reuse=True):
	temperature = 2./3.
	encoder_network = settings['architecture']['encoder']['fn']

	x_placeholder = tf_models.samples_placeholder()
	assert(not x_placeholder is None)

	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = encoder_network(settings, x_placeholder, is_training=False)
		dist_z_given_x = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature, logits=logits_z)
		logits_sample = tf.cast(dist_z_given_x.sample(), dtype=tf.float32)
		z_sample = tf.exp(logits_sample) * 2. - 1.

		encoder = tf.identity(z_sample, name='q_z_given_x/sample')
	return encoder

def create_decoder(settings, reuse=True):
	decoder_network = settings['architecture']['decoder']['fn']

	z_placeholder = tf_models.codes_placeholder()
	assert(not z_placeholder is None)

	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, tf.reshape(z_placeholder, (settings['batch_size'], -1)), is_training=False)
		#dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x, dtype=tf.float32)
		#decoder = tf.identity(dist_x_given_z.sample(), name='p_x_given_z/sample')
	#return decoder
	return tf.identity(tf.nn.sigmoid(logits_x), name='p_x_given_z/sample')

def create_probs(settings, inputs, is_training, reuse=False):
	temperature = 2./3.
	encoder_network = settings['architecture']['encoder']['fn']
	decoder_network = settings['architecture']['decoder']['fn']
	K = settings['count_categories']
	latent_batchshape = (settings['batch_size'], settings['latent_dimension'], K)
	
	#dist_prior = tf_models.standard_normal(tf_models.latentshape(settings))
	#dist_prior = tf.contrib.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
	temperature_prior = 0.5
	prior_prob = settings['prior_prob']
	logits_prior_prob = math.log(prior_prob / (1. - prior_prob))
	dist_prior = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature_prior, logits=tf.constant(0., shape=latent_batchshape))
	
	# Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = tf.reshape(encoder_network(settings, inputs, is_training=is_training), latent_batchshape)

	dist_z_given_x = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature, logits=logits_z)

	# Draw one sample z from Gumbel-softmax
	logits_sample = tf.cast(dist_z_given_x.sample(), dtype=tf.float32)

	# DEBUG
	#print('logits_sample.shape', logits_sample.shape)
	#raise Exception()

	# NOTE: Is this what is meant by "this running average was subtracted from the activity of the layer before it was updated"?
	#z_sample = tf.sigmoid(logits_sample) * 2. - 1. #- z_sample_avg
	z_sample = tf.exp(logits_sample) * 2. - 1.
	#z_sample = logits_sample

	# Use generator to determine mean of Bernoulli distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, tf.reshape(z_sample, (settings['batch_size'], -1)), is_training=is_training)

	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	
	# NOTE: x | z is defined as over each pixel separate, where prior on z is a multivariate
	# Hence the need to do the tf.reduce_sum op on the former to get down to a single number for each sample
	lg_p_x_given_z = tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(inputs)), 1, name='p_x_given_z/log_prob')

	lg_p_z = tf.identity(tf.reduce_sum(dist_prior.log_prob(logits_sample), 1), name='p_z/log_prob')
	lg_q_z_given_x = tf.identity(tf.reduce_sum(dist_z_given_x.log_prob(logits_sample), 1), name='q_z_given_x/log_prob')

	#print(dist_prior.log_prob(logits_sample).shape)
	#print(dist_z_given_x.log_prob(logits_sample).shape)
	#raise Exception()

	return lg_p_x_given_z, lg_p_z, lg_q_z_given_x

def lg_likelihood(x, z, settings, reuse=True, is_training=False):
	decoder_network = settings['architecture']['decoder']['fn']
	real_z = tf.exp(z)*2. - 1.

	with tf.variable_scope('model'):
		with tf.variable_scope('decoder', reuse=reuse):
			logits_x = decoder_network(settings, tf.reshape(real_z, (settings['batch_size'], -1)), is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	return tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(x)), 1)

def lg_prior(z, settings, reuse=True, is_training=False):
	temperature_prior = 0.5
	K = settings['count_categories']
	latent_batchshape = (settings['batch_size'], settings['latent_dimension'], K)
	
	dist_prior = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature_prior, logits=tf.constant(0., shape=latent_batchshape))
	return tf.reduce_sum(dist_prior.log_prob(tf.reshape(z, latent_batchshape)), 1)

def sample_prior(settings):
	temperature_prior = 0.5
	K = settings['count_categories']
	latent_batchshape = (settings['batch_size'], settings['latent_dimension'], K)
	
	dist_prior = tf.contrib.distributions.ExpRelaxedOneHotCategorical(temperature=temperature_prior, logits=tf.constant(0., shape=latent_batchshape))
	return tf.identity(tf.cast(tf.reshape(dist_prior.sample(), (settings['batch_size'], -1)), dtype=tf.float32), name='p_z/sample')
