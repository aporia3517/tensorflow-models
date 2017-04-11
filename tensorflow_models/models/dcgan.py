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

def create_placeholders(settings):
	#x = tf.placeholder(tf.float32, shape=tf_models.batchshape(settings), name='samples')
	z = tf.placeholder(tf.float32, shape=tf_models.latentshape(settings), name='codes')
	return z

def create_prior(settings):
	dist_prior = tf_models.gan_uniform()
	return tf.identity(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), name='p_z/sample')

def create_decoder(settings, reuse=True):
	z_placeholder = tf_models.codes_placeholder()
	assert(not z_placeholder is None)

	with tf.variable_scope('generator', reuse=reuse):
		decoder = tf.identity(generator_network(settings, z_placeholder), name='p_x/sample')
	return decoder

def generator_network(settings, code):
	# TODO: DC-GAN implemenation
	gf_dim = 64
	h = slim.fully_connected(code, gf_dim*2*7*7, scope='projection', activation_fn=tf.nn.elu)
	h = tf.reshape(h, [-1, 7, 7, gf_dim*2])
	h = slim.conv2d_transpose(h, gf_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.elu, scope='h2')
	h = slim.conv2d_transpose(h, 1, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid, scope='h3')
	h = tf.reshape(h, [-1, 28*28])

	return h

def discriminator_network(settings, inputs):
	# TODO: DC-GAN implementation
	h = tf.reshape(inputs, [100, 28, 28, 1])
	df_dim = 64
	h = slim.conv2d(h, df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.elu, scope='h1')
	h = slim.conv2d(h, 2*df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.elu, scope='h2')
	h = tf.reshape(h, [100, -1])
	h = slim.fully_connected(h, 1, activation_fn=tf.nn.sigmoid, scope='h3')

	return h

def create_probs(settings, inputs, reuse=False):
	eps = tf.random_uniform(tf_models.latentshape(settings), minval=-1., maxval=1., dtype=tf.float32)

	with tf.variable_scope('generator', reuse=reuse):
		fake = generator_network(settings, eps)

	with tf.variable_scope('discriminator', reuse=reuse):
		p_data = discriminator_network(settings, inputs)
		tf.get_variable_scope().reuse_variables()
		p_fake = discriminator_network(settings, fake)

	ll_data = tf.identity(tf.reduce_sum(tf_models.safe_log(p_data), 1), name='p_x/log_prob_real')
	ll_fake = tf.identity(tf.reduce_sum(tf_models.safe_log(p_fake), 1), name='p_x/log_prob_fake')
	ll_one_minus_fake = tf.identity(tf.reduce_sum(tf_models.safe_log(1. - p_fake), 1), name='p_x/log_one_minus_prob_fake')

	return ll_data, ll_fake, ll_one_minus_fake