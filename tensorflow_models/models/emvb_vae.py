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

# Black-box encoder: q(z | x, eps)
# Returns a sample from z given x and epsilon
def encoder_network(settings, inputs, eps, is_training):
	#with tf.variable_scope('q_z_given_x'):
	return tf_models.layers.mlp(
						tf.concat([inputs, eps], axis=1), 
						settings['encoder_sizes'] + [settings['latent_dimension']],
						final_activation_fn=tf.identity)

# Decoder: p(x | z)
# Returns parameters for bernoulli distribution on x given z
def decoder_network(settings, code, is_training):
	return tf_models.layers.bernoulli_parameters_mlp(code, settings['decoder_sizes'] + tf_models.flattened_shape(settings))

# Discriminator used for adversarial training in logits
def critic_network(settings, x, z, is_training):
	x_layer = tf_models.layers.mlp(x, settings['critic_x_sizes'], scope='x_layer')
	z_layer = tf_models.layers.mlp(z, settings['critic_z_sizes'], scope='z_layer')
	return tf_models.layers.mlp(
						tf.concat([x_layer, z_layer], axis=1),
						settings['critic_join_sizes'] + [1], scope='join_layer',
						final_activation_fn=tf.identity)

def create_encoder(settings, reuse=True):
	x_placeholder = tf_models.samples_placeholder()
	assert(not x_placeholder is None)

	noise = tf.random_normal(tf_models.noiseshape(settings), 0, 1, dtype=tf.float32)
	with tf.variable_scope('encoder', reuse=reuse):
		encoder = tf.identity(encoder_network(settings, x_placeholder, noise, is_training=False), name='q_z_given_x_eps/sample')
	return encoder

def create_decoder(settings, reuse=True):
	z_placeholder = tf_models.codes_placeholder()
	assert(not z_placeholder is None)

	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_placeholder, is_training=False)
		dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x)
		decoder = tf.identity(dist_x_given_z.sample(), name='p_x_given_z/sample')
	return decoder

def create_probs(settings, inputs, is_training, reuse=False):
	# The noise is distributed i.i.d. N(0, 1)
	noise = tf.random_normal(tf_models.noiseshape(settings), 0, 1, dtype=tf.float32)

	# Use black-box inference network to sample z, given inputs and noise
	with tf.variable_scope('encoder', reuse=reuse):
		z_sample = encoder_network(settings, inputs, noise, is_training=is_training)

	# The prior on z is also i.i.d. N(0, 1)
	z_prior = tf.random_normal(tf_models.latentshape(settings), 0, 1, dtype=tf.float32)
		
	# Use generator to determine distribution of reconstructed input
	with tf.variable_scope('decoder', reuse=reuse):
		logits_x = decoder_network(settings, z_sample, is_training=is_training)
	dist_x_given_z = tf.contrib.distributions.Bernoulli(logits=logits_x)

	# Log likelihood of reconstructed inputs
	lg_p_x_given_z = tf.identity(tf.reduce_sum(dist_x_given_z.log_prob(inputs), 1), name='p_x_given_z/log_prob')

	# Discriminator T(x, z)
	with tf.variable_scope('critic', reuse=reuse):
		critic = tf.identity(critic_network(settings, inputs, z_sample, is_training=is_training), name='generator')
		tf.get_variable_scope().reuse_variables()
		prior_critic = tf.identity(critic_network(settings, inputs, z_prior, is_training=is_training), name='prior')

	return lg_p_x_given_z, critic, prior_critic