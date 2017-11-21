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
import tensorflow_models.made
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
	K = settings['count_categories']

	x_placeholder = tf_models.samples_placeholder()
	assert(not x_placeholder is None)

	with tf.variable_scope('encoder', reuse=reuse):
		# Need to draw a sample from the encoder first, since the parameters of the MADE depend on the sample!
		index = tf.constant(0)
		condition = lambda i, s: tf.less(i, settings['latent_dimension'])

		logits_sample = tf.zeros([settings['batch_size'], settings['latent_dimension'], settings['count_categories']])

		uniform = tf.random_uniform(
			shape=[settings['batch_size'] * settings['latent_dimension'], settings['count_categories']],
			minval=np.finfo(np.float32).tiny,
			maxval=1.,
			dtype=tf.float32)	#,
			#seed=seed)

		gumbel_noise = -tf.log(-tf.log(uniform))

		def update(index, logits_sample):
			# Calculate the logits for the current 
			logits_z = tf_models.made.make_made_categorical(tf.reshape(logits_sample, [settings['batch_size'], -1]), x_placeholder, tf_masks, settings['latent_dimension'], 784, settings['hidden_dims'], count_categories=K, activation_fn=tf.tanh)
			
			made_dist = dist_fixed.ExpRelaxedOneHotCategorical(temperature=temperature, logits=logits_z, gumbel_noise=gumbel_noise)
			logits_sample = made_dist.sample()

			return tf.add(index, 1), logits_sample

		# TODO: Do I need to stop the gradients through the while_loop?
		_, logits_sample = tf.while_loop(condition, update, loop_vars=[index, logits_sample], back_prop=True, shape_invariants=[index.get_shape(), logits_sample.shape]) # swap_memory=True, parallel_iterations=1,

		
	return tf.identity(tf.exp(logits_sample) * 2.0 - 1., name='q_z_given_x/sample')

def create_decoder(settings, reuse=True):
	decoder_network = settings['architecture']['decoder']['fn']

	z_placeholder = tf_models.codes_placeholder()
	assert(not z_placeholder is None)

	with tf.variable_scope('decoder', reuse=reuse):
		# Need to draw a sample from the encoder first, since the parameters of the MADE depend on the sample!
		index = tf.constant(0)
		condition = lambda i, s: tf.less(i, 784)

		logits_sample = tf.zeros([settings['batch_size'], 784])

		uniform_noise = tf.random_uniform(
			shape=[1, settings['batch_size'], 784, 1],
			minval=np.finfo(np.float32).tiny,
			maxval=1.,
			dtype=tf.float32)

		def update(index, logits_sample):
			# Calculate the logits for the current 
			logits_x = tf_models.made.make_made_categorical(logits_sample, tf.reshape(z_placeholder, [settings['batch_size'], -1]), tf_masks_decoder, 784, settings['latent_dimension']*2, settings['hidden_dims_decoder'], count_categories=1, activation_fn=tf.tanh)
			
			made_dist = dist_fixed.Bernoulli(logits=logits_x, uniform_noise=uniform_noise, dtype=tf.float32)
			logits_sample = tf.reshape(made_dist.sample(), [settings['batch_size'], -1])

			return tf.add(index, 1), logits_sample

		# TODO: Do I need to stop the gradients through the while_loop?
		_, logits_sample = tf.while_loop(condition, update, loop_vars=[index, logits_sample], back_prop=True, shape_invariants=[index.get_shape(), logits_sample.shape])

		tf.get_variable_scope().reuse_variables()
		logits_x = tf_models.made.make_made_categorical(logits_sample, tf.reshape(z_placeholder, [settings['batch_size'], -1]), tf_masks_decoder, 784, settings['latent_dimension']*2, settings['hidden_dims_decoder'], count_categories=1, activation_fn=tf.tanh)

		#logits_x = decoder_network(settings, tf.reshape(z_placeholder, (settings['batch_size'], -1)), is_training=False)
		#dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x, dtype=tf.float32)
		#decoder = tf.identity(dist_x_given_z.sample(), name='p_x_given_z/sample')
	#return decoder
	return tf.identity(tf.nn.sigmoid(logits_x), name='p_x_given_z/sample')

# A hack to make this easy! Need to create masks inside the graph context mananger
tf_masks = None
masks = None

tf_masks_decoder = None
masks_decoder = None

def create_probs(settings, inputs, is_training, reuse=False):
	global tf_masks
	global masks
	global tf_masks_decoder
	global masks_decoder

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

	if masks is None:
		print('Creating TensorFlow MADE masks for the first time...')
		np.random.seed(seed=4)
		masks = tf_models.made.made_masks(settings['latent_dimension'], 784, settings['hidden_dims'], count_categories=K)
		masks_decoder = tf_models.made.made_masks(784, settings['latent_dimension']*2, settings['hidden_dims_decoder'])
	else:
		print('Reusing existing TensorFlow MADE masks...')

	if not reuse:
		tf_masks = [tf.constant(m.transpose(), dtype=tf.float32) for m in masks]
		tf_masks_decoder = [tf.constant(m.transpose(), dtype=tf.float32) for m in masks_decoder]

	# DEBUG
	#print([m.shape for m in masks])
	#raise Exception()
	
	# Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
	with tf.variable_scope('encoder', reuse=reuse):
		# Need to draw a sample from the encoder first, since the parameters of the MADE depend on the sample!
		index = tf.constant(0)
		condition = lambda i, s: tf.less(i, settings['latent_dimension'])

		logits_sample = tf.zeros([settings['batch_size'], settings['latent_dimension'], settings['count_categories']])

		uniform = tf.random_uniform(
			shape=[settings['batch_size'] * settings['latent_dimension'], settings['count_categories']],
			minval=np.finfo(np.float32).tiny,
			maxval=1.,
			dtype=tf.float32)	#,
			#seed=seed)

		gumbel_noise = -tf.log(-tf.log(uniform))

		# DEBUG
		#print('logits_sample.shape', logits_sample.shape)

		def update(index, logits_sample):
			# Calculate the logits for the current 
			logits_z = tf_models.made.make_made_categorical(tf.reshape(logits_sample, [settings['batch_size'], -1]), inputs, tf_masks, settings['latent_dimension'], 784, settings['hidden_dims'], count_categories=K, activation_fn=tf.tanh)
			
			made_dist = dist_fixed.ExpRelaxedOneHotCategorical(temperature=temperature, logits=logits_z, gumbel_noise=gumbel_noise)
			logits_sample = made_dist.sample()

			return tf.add(index, 1), logits_sample

		# TODO: Do I need to stop the gradients through the while_loop?
		_, logits_sample = tf.while_loop(condition, update, loop_vars=[index, logits_sample], back_prop=True, shape_invariants=[index.get_shape(), logits_sample.shape]) # swap_memory=True, parallel_iterations=1

		# Use the sample from the MADE to evaluate the final logitistic parameters
		tf.get_variable_scope().reuse_variables()
		logits_z = tf_models.made.make_made_categorical(tf.reshape(logits_sample, [settings['batch_size'], -1]), inputs, tf_masks, settings['latent_dimension'], 784, settings['hidden_dims'], count_categories=K, activation_fn=tf.tanh)

	dist_z_given_x = dist_fixed.ExpRelaxedOneHotCategorical(temperature=temperature, logits=logits_z, gumbel_noise=gumbel_noise)

	# DEBUG
	#print('logits_sample.shape', logits_sample.shape)
	#print('logits_z.shape', logits_z.shape)
	#raise Exception()

	# NOTE: Is this what is meant by "this running average was subtracted from the activity of the layer before it was updated"?
	#z_sample = tf.sigmoid(logits_sample) * 2. - 1. #- z_sample_avg
	z_sample = tf.exp(logits_sample) * 2. - 1.
	#z_sample = logits_sample

	# Use generator to determine mean of Bernoulli distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = tf_models.made.make_made_categorical(inputs, tf.reshape(z_sample, [settings['batch_size'], -1]), tf_masks_decoder, 784, settings['latent_dimension']*2, settings['hidden_dims_decoder'], count_categories=1, activation_fn=tf.tanh)

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

"""def lg_likelihood(x, z, settings, reuse=True, is_training=False):
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
	return tf.identity(tf.cast(tf.reshape(dist_prior.sample(), (settings['batch_size'], -1)), dtype=tf.float32), name='p_z/sample')"""

def lg_likelihood(x, z, settings, reuse=True, is_training=False):
	#decoder_network = settings['architecture']['decoder']['fn']
	latent_batchshape = (settings['batch_size'], settings['latent_dimension'], 1)
	real_z1 = tf.reshape(tf.sigmoid(z)*2. - 1., latent_batchshape)
	real_z2 = tf.reshape((1.-tf.sigmoid(z))*2. - 1., latent_batchshape)
	real_z = tf.reshape(tf.concat([real_z1, real_z2], axis=2), (settings['batch_size'],-1))

	with tf.variable_scope('model'):
		with tf.variable_scope('decoder', reuse=reuse):
			logits_x = tf_models.made.make_made_categorical(x, real_z, tf_masks_decoder, 784, settings['latent_dimension'], settings['hidden_dims_decoder'], count_categories=1, activation_fn=tf.tanh)
			
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
