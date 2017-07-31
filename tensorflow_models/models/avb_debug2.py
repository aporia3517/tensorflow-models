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

def create_placeholders(settings):
	x = tf.placeholder(tf.float32, shape=tf_models.batchshape(settings), name='samples')
	z = tf.placeholder(tf.float32, shape=tf_models.latentshape(settings), name='codes')
	return x, z

def create_prior(settings):
	temperature = 0.5
	prior_prob = settings['prior_prob']
	dist_prior = tf.contrib.distributions.RelaxedBernoulli(temperature, probs=prior_prob)
	return tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32)*2. - 1., name='p_z/sample')

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

	# Create a tiled version of the inputs for adaptive contrast
	inputs_ac = tf.tile(tf.expand_dims(tf_models.flatten(inputs), axis=0), multiples=[settings['ac_size'],1,1])
	noise_ac = tf.random_normal((settings['ac_size'], settings['batch_size'], settings['noise_dimension']), 0, 1, dtype=tf.float32)

	#print('inputs_ac.shape', inputs_ac.shape)
	#print('noise_ac.shape', noise_ac.shape)

	# Use black-box inference network to sample z, given inputs and noise
	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = encoder_network(settings, inputs, noise, is_training=is_training)
		tf.get_variable_scope().reuse_variables()
		logits_z_ac = encoder_network(settings, tf.reshape(inputs_ac, (settings['ac_size']*settings['batch_size'], -1)), tf.reshape(noise_ac, (settings['ac_size']*settings['batch_size'], -1)), is_training=is_training)
		logits_z_ac = tf.reduce_mean(tf.reshape(logits_z_ac, (settings['ac_size'], settings['batch_size'], -1)), 0)

		#print('logits_z_ac.shape', logits_z_ac.shape)
		#raise Exception()
	
	#dist_z_given_x = tf.contrib.distributions.Logistic(loc=logits_z/temperature, scale=tf.constant(1./temperature, shape=logits_z.shape))
	#logits_sample = tf.identity(tf.cast(dist_z_given_x.sample(), dtype=tf.float32), name='z/sample')
	#z_sample = tf.identity(tf.sigmoid(logits_sample) * 2. - 1.)

	dist_z_given_x_ac = tf.contrib.distributions.Logistic(loc=logits_z_ac/temperature, scale=tf.constant(1./temperature, shape=logits_z_ac.shape))
	logits_sample_ac = tf.identity(tf.cast(dist_z_given_x_ac.sample(), dtype=tf.float32))
	z_sample_ac = tf.identity(tf.sigmoid(logits_sample_ac) * 2. - 1.)

	dist_prior_ac = tf.contrib.distributions.Logistic(loc=0., scale=1.)
	sample_prior_ac = tf.identity(tf.cast(dist_prior_ac.sample(sample_shape=(settings['batch_size'], settings['latent_dimension'])), dtype=tf.float32), name='z/prior')

	dist_z_given_x = tf.contrib.distributions.Logistic(loc=logits_z/temperature, scale=tf.constant(1./temperature, shape=logits_z.shape))
	logits_sample = tf.identity(tf.cast(dist_z_given_x.sample(), dtype=tf.float32)) #, name='z/sample')
	z_sample = tf.identity(tf.sigmoid(logits_sample) * 2. - 1.)

	sample_for_discr = tf.identity(temperature * logits_sample - logits_z_ac, name='z/sample')
	#sample_for_discr = z_sample

	#print(logits_sample._shape, logits_z_ac.shape, sample_prior_ac.shape)

	#print(logits_sample.shape)
	#print(z_sample.shape)

	# The prior on z is also i.i.d. N(0, 1)
	temperature_prior = 0.5
	prior_prob = settings['prior_prob']
	logits_prior_prob = math.log(prior_prob / (1. - prior_prob))

	# TODO: Have to check this line!
	dist_prior = tf.contrib.distributions.Logistic(loc=logits_prior_prob/temperature_prior, scale=1./temperature_prior)

	logits_prior = tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32))
	z_prior = tf.identity(tf.sigmoid(logits_prior)*2. - 1.)

	#sample_prior_ac = z_sample_ac

	#print(logits_prior.shape)
	#print(z_prior.shape)
	#raise Exception()
		
	# Use generator to determine distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_sample, is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)

	# Log likelihood of reconstructed inputs
	#probs = dist_x_given_z.log_prob(tf_models.flatten(inputs))
	#print(probs.shape)

	lg_p_x_given_z = tf.identity(tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(inputs)), 1), name='p_x_given_z/log_prob')

	lg_r_alpha = tf.identity(tf.reduce_sum(dist_z_given_x_ac.log_prob(logits_sample), 1), name='r_alpha/log_prob')

	#print(lg_p_x_given_z.shape)
	#raise Exception()

	# Discriminator T(x, z)
	with tf.variable_scope('discriminator', reuse=reuse):
		discriminator = tf.squeeze(discriminator_network(settings, inputs, sample_for_discr, is_training=is_training), name='generator')
		tf.get_variable_scope().reuse_variables()
		prior_discriminator = tf.squeeze(discriminator_network(settings, inputs, sample_prior_ac, is_training=is_training), name='prior')

	#print(lg_p_x_given_z.shape)
	#print(discriminator.shape)
	#print(prior_discriminator.shape)
	#raise Exception()

	lg_p_z = tf.identity(tf.reduce_sum(dist_prior.log_prob(logits_sample), 1), name='p_z/log_prob')
	#lg_q_z_given_x = tf.identity(tf.reduce_sum(dist_z_given_x.log_prob(logits_sample), 1), name='q_z_given_x/log_prob')

	return lg_p_x_given_z, discriminator, prior_discriminator, lg_p_z, lg_r_alpha, sample_for_discr, sample_prior_ac  #lg_q_z_given_x #, z_sample, z_prior, logits_sample, logits_prior,

def lg_likelihood(x, z, settings, reuse=True, is_training=False):
	decoder_network = settings['architecture']['decoder']['fn']
	real_z = tf.sigmoid(z)*2. - 1.

	with tf.variable_scope('model'):
		with tf.variable_scope('decoder', reuse=reuse):
			logits_x = decoder_network(settings, real_z, is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	return tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(x)), 1)

def lg_prior(z, settings, reuse=True, is_training=False):
	temperature = 0.5
	prior_prob = settings['prior_prob']
	logits_prior_prob = math.log(prior_prob / (1. - prior_prob))
	dist_prior = tf.contrib.distributions.Logistic(loc=logits_prior_prob/temperature, scale=1./temperature)
	return tf.reduce_sum(tf_models.flatten(dist_prior.log_prob(z)), 1)

def sample_prior(settings):
	temperature = 0.5
	prior_prob = settings['prior_prob']
	logits_prior_prob = math.log(prior_prob / (1. - prior_prob))
	dist_prior = tf.contrib.distributions.Logistic(loc=logits_prior_prob/temperature, scale=1./temperature)
	return tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32), name='p_z/sample')
