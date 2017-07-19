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

# Adversarial variational Bayes loss

# lg_p_x_given_z ~ batch_size x 784
# adversary ~ ?
# prior_adversary ~ ?
def loss(lg_p_x_given_z, discriminator, prior_discriminator, x, z_sample, z_prior, name, bandwidth):	
	# Eq (3.9)
	# NOTE: Take negative since we are minimizing
	elbo_loss = -tf.reduce_mean(-discriminator + lg_p_x_given_z)

	grad = tf.gradients(discriminator, [z_sample])
	grad2 = tf.gradients(discriminator, [x])
	grad_norm_fake = tf.reduce_sum(tf.square(tf.concat([grad[0], tf_models.flatten(grad2[0])], axis=1)), axis=1)


	grad = tf.gradients(prior_discriminator, [z_prior])
	grad2 = tf.gradients(prior_discriminator, [x])
	grad_norm_prior = tf.reduce_sum(tf.square(tf.concat([grad[0], tf_models.flatten(grad2[0])], axis=1)), axis=1)

	penalty = tf.square(1. - tf.nn.sigmoid(discriminator)) * grad_norm_fake + tf.square(tf.nn.sigmoid(prior_discriminator)) * grad_norm_prior
	#print('penalty.shape', penalty.shape)

	# Eq (3.3)
	discriminator_loss = -tf.reduce_mean(tf_models.safe_log(tf.nn.sigmoid(discriminator)) + tf_models.safe_log(1. - tf.nn.sigmoid(prior_discriminator)))
	#discriminator_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(discriminator)) + tf.log(1. - tf.nn.sigmoid(prior_discriminator)))

	if bandwidth != 0.:
		discriminator_loss += tf.reduce_mean(penalty) * bandwidth / 2.

	return tf.identity(elbo_loss, name=name+'/elbo_like'), tf.identity(discriminator_loss, name=name+'/discriminator')

def create(name='train', settings=None):
	lg_p_x_given_z = tf.squeeze(tf_models.get_output(name + '/p_x_given_z/log_prob'))
	discriminator = tf.squeeze(tf_models.get_output(name + '/discriminator/generator'))
	prior_discriminator = tf.squeeze(tf_models.get_output(name + '/discriminator/prior'))

	x = get_input(name)
	z_sample = tf_models.get_output(name + '/z/sample')
	z_prior = tf_models.get_output(name + '/z/prior')

	#print('lg_p_x_given_z.shape', lg_p_x_given_z.shape)
	#print('discriminator.shape', discriminator.shape)
	#print('prior_discriminator.shape', prior_discriminator.shape)
	#raise Exception()

	return loss(lg_p_x_given_z, discriminator, prior_discriminator, x, z_sample, z_prior, name=name, bandwidth=settings['avb_bandwidth'])

def get_input(name):
	ops = tf.get_collection(tf_models.GraphKeys.INPUTS)
	for op in ops:
		if 'inputs/'+name+'/samples' in op.name:
			return op
	raise ValueError('No loss operation with substring "{}" exists'.format(name))
