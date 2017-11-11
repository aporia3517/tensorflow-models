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

#def create_prior(settings):
#	dist_prior = tf.contrib.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
#	return tf.identity(dist_prior.sample(sample_shape=tf_models.latentshape(settings)) * 2. - 1., name='p_z/sample')

def create_prior(settings):
	temperature = 0.5
	prior_prob = settings['prior_prob']
	dist_prior = tf.contrib.distributions.RelaxedBernoulli(temperature, probs=prior_prob)
	return tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32) * 2. - 1., name='p_z/sample')

def create_encoder(settings, reuse=True):
	temperature = 2./3.
	encoder_network = settings['architecture']['encoder']['fn']

	x_placeholder = tf_models.samples_placeholder()
	assert(not x_placeholder is None)

	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = encoder_network(settings, x_placeholder, is_training=False)
		dist_z_given_x = tf.contrib.distributions.RelaxedBernoulli(temperature, logits=logits_z)
		encoder = tf.identity(tf.cast(dist_z_given_x.sample(name='sample'), dtype=tf.float32) * 2. - 1., name='q_z_given_x/sample')
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
	
	#dist_prior = tf_models.standard_normal(tf_models.latentshape(settings))
	#dist_prior = tf.contrib.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
	temperature_prior = 0.5
	prior_prob = settings['prior_prob']
	logits_prior_prob = math.log(prior_prob / (1. - prior_prob))
	dist_prior = tf.contrib.distributions.Logistic(loc=logits_prior_prob/temperature_prior, scale=1./temperature_prior)
	dist_prior_discrete = tf.contrib.distributions.Bernoulli(probs=prior_prob, dtype=tf.float32)
	#dist_prior = tf.contrib.distributions.RelaxedBernoulli(temperature_prior, probs=0.5)

	#with tf.variable_scope('centering', reuse=reuse):
	#	z_sample_avg = tf.get_variable('z_sample_avg', shape=tf_models.latentshape(settings), initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=False)

	# Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
	with tf.variable_scope('encoder', reuse=reuse):
		logits_z = encoder_network(settings, inputs, is_training=is_training)
		
	#dist_z_given_x = tf.contrib.distributions.RelaxedBernoulli(temperature, logits=logits_z)
	dist_z_given_x = tf.contrib.distributions.Logistic(loc=logits_z/temperature, scale=tf.constant(1./temperature, shape=logits_z.shape))
	dist_z_given_x_discrete = tf.contrib.distributions.Bernoulli(logits=logits_z, dtype=tf.float32)

	# Draw one sample z from Gumbel-softmax
	logits_sample = tf.cast(dist_z_given_x.sample(), dtype=tf.float32)

	# NOTE: Is this what is meant by "this running average was subtracted from the activity of the layer before it was updated"?
	z_sample = tf.sigmoid(logits_sample) * 2. - 1. #- z_sample_avg
	z_sample_discrete = tf.round(z_sample)

	# Create moving average ops on first creation
	#decay = 0.9
	#if not reuse:
	#	update_z_sample_avg = z_sample_avg.assign_sub((1 - decay) * (z_sample_avg - z_sample))
	#	tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_z_sample_avg)

	#print('z_sample.shape', z_sample.shape)

	# Use generator to determine mean of Bernoulli distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_sample, is_training=is_training)
		tf.get_variable_scope().reuse_variables()
		logits_x_discrete = decoder_network(settings, z_sample_discrete, is_training=is_training)

	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)
	dist_x_given_z_discrete = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x_discrete), dtype=tf.float32)
	
	# NOTE: x | z is defined as over each pixel separate, where prior on z is a multivariate
	# Hence the need to do the tf.reduce_sum op on the former to get down to a single number for each sample
	lg_p_x_given_z = tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(inputs)), 1, name='p_x_given_z/log_prob')

	lg_p_z = tf.identity(tf.reduce_sum(dist_prior.log_prob(logits_sample), 1), name='p_z/log_prob')
	lg_q_z_given_x = tf.identity(tf.reduce_sum(dist_z_given_x.log_prob(logits_sample), 1), name='q_z_given_x/log_prob')

	lg_p_z_discrete = tf.identity(tf.reduce_sum(dist_prior_discrete.log_prob(z_sample_discrete), 1), name='p_z/log_prob_discrete')
	lg_q_z_given_x_discrete = tf.identity(tf.reduce_sum(dist_z_given_x_discrete.log_prob(z_sample_discrete), 1), name='q_z_given_x/log_prob_discrete')
	lg_p_x_given_z_discrete = tf.reduce_sum(dist_x_given_z_discrete.log_prob(tf_models.flatten(inputs)), 1, name='p_x_given_z/log_prob_discrete')

	return lg_p_x_given_z, lg_p_z, lg_q_z_given_x, lg_p_z_discrete, lg_q_z_given_x_discrete, lg_p_x_given_z_discrete

def lg_likelihood(x, z, settings, reuse=True, is_training=False):
	decoder_network = settings['architecture']['decoder']['fn']
	real_z = tf.sigmoid(z)*2. - 1.

	with tf.variable_scope('model'):
		with tf.variable_scope('decoder', reuse=reuse):
			logits_x = decoder_network(settings, real_z, is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=tf_models.flatten(logits_x), dtype=tf.float32)

	#print('lg_likelihood.shape', tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(x)), 1).shape)

	return tf.reduce_sum(dist_x_given_z.log_prob(tf_models.flatten(x)), 1)

def lg_prior(z, settings, reuse=True, is_training=False):
	temperature = 0.5
	prior_prob = settings['prior_prob']
	logits_prior_prob = math.log(prior_prob / (1. - prior_prob))
	dist_prior = tf.contrib.distributions.Logistic(loc=logits_prior_prob/temperature, scale=1./temperature)

	#print('lg_prior.shape', tf.reduce_sum(tf_models.flatten(dist_prior.log_prob(z)), 1).shape)

	return tf.reduce_sum(tf_models.flatten(dist_prior.log_prob(z)), 1)

def sample_prior(settings):
	temperature = 0.5
	prior_prob = settings['prior_prob']
	logits_prior_prob = math.log(prior_prob / (1. - prior_prob))
	dist_prior = tf.contrib.distributions.Logistic(loc=logits_prior_prob/temperature, scale=1./temperature)

	#print('sample_prior.shape', tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32).shape)

	return tf.identity(tf.cast(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), dtype=tf.float32), name='p_z/sample')