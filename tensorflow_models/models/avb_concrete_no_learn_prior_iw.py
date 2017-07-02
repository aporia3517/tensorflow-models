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
	#temperature = 0.5
	#dist_prior = tf.contrib.distributions.RelaxedBernoulli(temperature, probs=0.5)
	temperature_prior = 0.5
	dist_prior = tf.contrib.distributions.Logistic(loc=0., scale=1./temperature_prior)
	#return tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32)*2. - 1., name='p_z/sample')
	return tf.identity(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), name='p_z/sample')

def create_encoder(settings, reuse=True):
	encoder_network = settings['architecture']['encoder']['fn']
	temperature = 2./3.

	x_placeholder = tf_models.samples_placeholder()
	assert(not x_placeholder is None)

	noise = tf.random_normal(tf_models.noiseshape(settings), 0, 1, dtype=tf.float32)
	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = encoder_network(settings, x_placeholder, noise, is_training=False)
		#dist_z_given_x = tf.contrib.distributions.RelaxedBernoulli(temperature, logits=logits_z)
		#encoder = tf.identity(tf.cast(dist_z_given_x.sample(), dtype=tf.float32) * 2. - 1., name='q_z_given_x_eps/sample')

		dist_z_given_x = tf.contrib.distributions.Logistic(loc=logits_z/temperature, scale=1./temperature)
		encoder = tf.identity(tf.cast(dist_z_given_x.sample(), dtype=tf.float32), name='q_z_given_x_eps/sample')

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
	iw_shape = (settings['iw_size'], settings['batch_size'], -1)
	flat_shape = (-1, settings['latent_dimension'])

	encoder_network = settings['architecture']['encoder']['fn']
	decoder_network = settings['architecture']['decoder']['fn']
	discriminator_network = settings['architecture']['adversary']['fn']

	# The noise is distributed i.i.d. N(0, 1)
	noise = tf.random_normal(tf_models.noiseshape(settings), 0, 1, dtype=tf.float32)

	# Use black-box inference network to sample z, given inputs and noise
	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = encoder_network(settings, inputs, noise, is_training=is_training)
		dist_z_given_x = tf.contrib.distributions.Logistic(loc=logits_z/temperature, scale=tf.constant(1./temperature, shape=logits_z.shape))
		logits_sample = tf.cast(dist_z_given_x.sample(), dtype=tf.float32)
		logits_sample_iw = tf.cast(dist_z_given_x.sample([settings['iw_size']]), dtype=tf.float32)
		z_sample = tf.sigmoid(logits_sample)*2. - 1.
		z_sample_iw = tf.sigmoid(logits_sample_iw)*2. - 1.

	# The prior on z is also i.i.d. N(0, 1)
	#dist_prior = tf.contrib.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
	temperature_prior = 0.5
	dist_prior = tf.contrib.distributions.Logistic(loc=0., scale=1./temperature_prior)
	logits_prior = dist_prior.sample(sample_shape=tf_models.latentshape(settings))
	logits_prior_iw = dist_prior.sample(sample_shape=(settings['iw_size'], settings['batch_size'], settings['latent_dimension']))
	z_prior = tf.sigmoid(logits_prior)*2. - 1.
	z_prior_iw = tf.sigmoid(logits_prior_iw)*2. - 1.

	# Calculate the quantities we need when only learning the lg(q(z|x)) term in the optimal discriminator, T*
	lg_p_z = tf.identity(tf.reduce_sum(dist_prior.log_prob(logits_sample), 1), name='p_z/log_prob')
	lg_p_z_iw = tf.identity(tf.reduce_sum(dist_prior.log_prob(logits_sample_iw), 2), name='p_z_iw/log_prob')

	lg_p_z_prior = tf.identity(tf.reduce_sum(dist_prior.log_prob(logits_prior), 1), name='p_z/log_prob_prior')
	lg_p_z_prior_iw = tf.identity(tf.reduce_sum(dist_prior.log_prob(logits_prior_iw), 2), name='p_z_iw/log_prob_prior')

	#print('lg_p_z_prior.shape', lg_p_z_prior.shape)
	#print('lg_p_z_prior_iw.shape', lg_p_z_prior_iw.shape)
	#raise Exception()
		
	# Use generator to determine distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		#logits_x = decoder_network(settings, z_sample, is_training=is_training)
		logits_x = decoder_network(settings, logits_sample, is_training=is_training)
		tf.get_variable_scope().reuse_variables()
		#logits_x_iw = decoder_network(settings, tf.reshape(z_sample_iw, flat_shape), is_training=is_training)
		logits_x_iw = decoder_network(settings, tf.reshape(logits_sample_iw, flat_shape), is_training=is_training)

	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	dist_x_given_z_iw = tf.contrib.distributions.Bernoulli(logits=tf.reshape(tf_models.flatten(logits_x_iw), iw_shape), dtype=tf.float32)

	# Log likelihood of reconstructed inputs
	lg_p_x_given_z = tf.identity(tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(inputs)), 1), name='p_x_given_z/log_prob')

	inputs_iw = tf.tile(tf.expand_dims(tf_models.flatten(inputs), axis=0), multiples=[settings['iw_size'],1,1])
	flat_inputs_iw = tf.reshape(inputs_iw, (settings['iw_size']*settings['batch_size'], -1))
	lg_p_x_given_z_iw = tf.reduce_sum(dist_x_given_z_iw.log_prob(inputs_iw), 2, name='p_x_given_z_iw/log_prob')

	# Discriminator T(x, z)
	with tf.variable_scope('discriminator', reuse=reuse):
		#discriminator = tf.squeeze(discriminator_network(settings, inputs, z_sample, is_training=is_training), name='generator')
		#discriminator = tf.identity(tf.squeeze(discriminator_network(settings, inputs, z_sample, is_training=is_training)) - lg_p_z, name='generator')
		discriminator = tf.identity(tf.squeeze(discriminator_network(settings, inputs, logits_sample, is_training=is_training)) - lg_p_z, name='generator')
		tf.get_variable_scope().reuse_variables()
		#prior_discriminator = tf.squeeze(discriminator_network(settings, inputs, z_prior, is_training=is_training), name='prior')
		#prior_discriminator = tf.identity(tf.squeeze(discriminator_network(settings, inputs, z_prior, is_training=is_training)) - lg_p_z_prior, name='prior')
		prior_discriminator = tf.identity(tf.squeeze(discriminator_network(settings, inputs, logits_prior, is_training=is_training)) - lg_p_z_prior, name='prior')

		flat_z_sample_iw = tf.reshape(z_sample_iw, (-1, settings['latent_dimension']))
		flat_logits_sample_iw = tf.reshape(logits_sample_iw, (-1, settings['latent_dimension']))
		#flat_z_prior_iw = tf.reshape(z_prior_iw, (-1, settings['latent_dimension']))

		#discriminator_iw = tf.squeeze(tf.reshape(discriminator_network(settings, flat_inputs_iw, flat_z_sample_iw, is_training=is_training), iw_shape), name='generator_iw')
		#discriminator_iw = tf.identity(tf.squeeze(tf.reshape(discriminator_network(settings, flat_inputs_iw, flat_z_sample_iw, is_training=is_training), iw_shape)) - lg_p_z_iw, name='generator_iw')
		discriminator_iw = tf.identity(tf.squeeze(tf.reshape(discriminator_network(settings, flat_inputs_iw, flat_logits_sample_iw, is_training=is_training), iw_shape)) - lg_p_z_iw, name='generator_iw')
		#prior_discriminator_iw = tf.squeeze(tf.reshape(discriminator_network(settings, flat_inputs_iw, flat_z_prior_iw, is_training=is_training), iw_shape), name='prior_iw')

		#(settings['iw_size'], settings['batch_size'], -1)

	# DEBUG: Check shapes
	"""print('discriminator.shape', discriminator.shape)
	print('prior_discriminator.shape', prior_discriminator.shape)
	print('lg_p_x_given_z.shape', lg_p_x_given_z.shape)
	print('discriminator_iw.shape', discriminator_iw.shape)
	print('prior_discriminator_iw.shape', prior_discriminator_iw.shape)
	print('lg_p_x_given_z_iw.shape', lg_p_x_given_z_iw.shape)"""

	return lg_p_x_given_z, discriminator, prior_discriminator, lg_p_x_given_z_iw, discriminator_iw #, lg_p_z, lg_p_z_iw, lg_p_z_prior, lg_p_z_prior_iw #, prior_discriminator_iw

def lg_likelihood(x, z, settings, reuse=True, is_training=False):
	decoder_network = settings['architecture']['decoder']['fn']
	#real_z = tf.sigmoid(z)*2. - 1.
	real_z = z

	with tf.variable_scope('model'):
		with tf.variable_scope('decoder', reuse=reuse):
			logits_x = decoder_network(settings, real_z, is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	return tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(x)), 1)

def lg_prior(z, settings, reuse=True, is_training=False):
	temperature = 0.5
	dist_prior = tf.contrib.distributions.Logistic(loc=0., scale=1./temperature)
	return tf.reduce_sum(tf_models.flatten(dist_prior.log_prob(z)), 1)

def sample_prior(settings):
	temperature = 0.5
	dist_prior = tf.contrib.distributions.Logistic(loc=0., scale=1./temperature)
	return tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32), name='p_z/sample')
