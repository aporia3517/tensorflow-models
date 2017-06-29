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
	temperature = 0.5
	dist_prior = tf.contrib.distributions.RelaxedBernoulli(temperature, probs=0.5)
	return tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32) * 2. - 1., name='p_z/sample')

def create_encoder(settings, reuse=True):
	encoder_network = settings['architecture']['encoder']['fn']
	temperature = 2./3.

	x_placeholder = tf_models.samples_placeholder()
	assert(not x_placeholder is None)

	noise = tf.random_normal(tf_models.noiseshape(settings), 0, 1, dtype=tf.float32)
	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = encoder_network(settings, x_placeholder, noise, is_training=False)
		dist_z_given_x = tf.contrib.distributions.RelaxedBernoulli(temperature, logits=logits_z)
		encoder = tf.identity(tf.cast(dist_z_given_x.sample(), dtype=tf.float32) * 2. - 1., name='q_z_given_x_eps/sample')
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
	temperature = 2./3.

	encoder_network = settings['architecture']['encoder']['fn']
	decoder_network = settings['architecture']['decoder']['fn']
	discriminator_network = settings['architecture']['adversary']['fn']

	# The noise is distributed i.i.d. N(0, 1)
	noise = tf.random_normal(tf_models.noiseshape(settings), 0, 1, dtype=tf.float32)

	# Use black-box inference network to sample z, given inputs and noise
	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = encoder_network(settings, inputs, noise, is_training=is_training)
		dist_z_given_x = tf.contrib.distributions.RelaxedBernoulli(temperature, logits=logits_z)
		z_sample = tf.cast(dist_z_given_x.sample(), dtype=tf.float32)*2. - 1.

	# The prior on z is also i.i.d. N(0, 1)
	#dist_prior = tf.contrib.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
	temperature_prior = 0.5
	dist_prior = tf.contrib.distributions.RelaxedBernoulli(temperature_prior, probs=0.5)
	z_prior = dist_prior.sample(sample_shape=tf_models.latentshape(settings))*2. - 1.
		
	# Use generator to determine distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_sample, is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)

	# Log likelihood of reconstructed inputs
	lg_p_x_given_z = tf.identity(tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(inputs)), 1), name='p_x_given_z/log_prob')

	# Discriminator T(x, z)
	with tf.variable_scope('discriminator', reuse=reuse):
		discriminator = tf.identity(discriminator_network(settings, inputs, z_sample, is_training=is_training), name='generator')
		tf.get_variable_scope().reuse_variables()
		prior_discriminator = tf.identity(discriminator_network(settings, inputs, z_prior, is_training=is_training), name='prior')

	return lg_p_x_given_z, discriminator, prior_discriminator

def lg_likelihood(x, z, settings, reuse=True, is_training=False):
	decoder_network = settings['architecture']['decoder']['fn']
	real_z = tf.sigmoid(z)*2. - 1.

	with tf.variable_scope('model'):
		with tf.variable_scope('decoder', reuse=reuse):
			logits_x = decoder_network(settings, real_z, is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	return tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(x)), 1)

"""def lg_prior(z, settings, reuse=True, is_training=False):
	dist_prior = tf.contrib.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
	return tf.reduce_sum(tf_models.flatten(dist_prior.log_prob((z + 1.)/2.)), 1)"""

def lg_prior(z, settings, reuse=True, is_training=False):
	temperature = 0.5
	dist_prior = tf.contrib.distributions.Logistic(loc=0., scale=1./temperature)
	return tf.reduce_sum(tf_models.flatten(dist_prior.log_prob(z)), 1)

def sample_prior(settings):
	temperature = 0.5
	dist_prior = tf.contrib.distributions.Logistic(loc=0., scale=1./temperature)
	return tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32), name='p_z/sample')
