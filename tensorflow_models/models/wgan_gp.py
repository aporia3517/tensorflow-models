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
	z = tf.placeholder(tf.float32, shape=tf_models.latentshape(settings), name='codes')
	return z

def create_prior(settings):
	dist_prior = tf_models.gan_uniform()
	return tf.identity(dist_prior.sample(sample_shape=tf_models.latentshape(settings)), name='p_z/sample')

def create_decoder(settings, reuse=True):
	generator_network = settings['architecture']['decoder']['fn']

	z_placeholder = tf_models.codes_placeholder()
	assert(not z_placeholder is None)

	with tf.variable_scope('generator', reuse=reuse):
		decoder = tf.identity(generator_network(settings, z_placeholder, is_training=False), name='p_x/sample')
	return decoder

def create_probs(settings, inputs, is_training, reuse=False):
	generator_network = settings['architecture']['decoder']['fn']
	critic_network = settings['architecture']['adversary']['fn']

	eps = tf.random_uniform(tf_models.latentshape(settings), minval=-1., maxval=1., dtype=tf.float32)

	with tf.variable_scope('generator', reuse=reuse):
		fake = generator_network(settings, eps, is_training=is_training)

	#print(type(tf_models.batchshape(settings)), len())

	eps = tf.random_uniform([settings['batch_size']] + [1]*(len(tf_models.batchshape(settings)) - 1), minval=0., maxval=1.)

	interpolated = tf.identity(eps*inputs + (1. - eps)*fake, name='x/interpolated')

	with tf.variable_scope('critic', reuse=reuse):
		p_data = critic_network(settings, inputs, is_training=is_training)
		tf.get_variable_scope().reuse_variables()
		p_fake = critic_network(settings, fake, is_training=is_training)
		p_interpolated = critic_network(settings, interpolated, is_training=is_training)

	critic_real = tf.identity(tf.reduce_sum(p_data, 1), name='p_x/critic_real')
	critic_fake = tf.identity(tf.reduce_sum(p_fake, 1), name='p_x/critic_fake')
	critic_interpolated = tf.identity(p_interpolated, name='p_x/critic_interpolated')

	return critic_real, critic_fake, critic_interpolated, interpolated
