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
def loss(lg_p_x_given_z, discriminator, prior_discriminator, lg_p_x_given_z_iw, discriminator_iw, lg_p_z, lg_p_z_iw, lg_p_z_prior, lg_p_z_iw_prior, name):	
	# DEBUG: Check shapes
	#print(discriminator.shape, lg_p_z.shape, lg_p_x_given_z.shape)
	#print(prior_discriminator.shape, lg_p_z_prior.shape)
	#print(lg_p_x_given_z_iw.shape, discriminator_iw.shape, lg_p_z_iw.shape)
	#raise Exception()

	# Eq (3.3)
	discriminator_loss = -tf.reduce_mean(tf_models.safe_log(tf.nn.sigmoid(discriminator - lg_p_z)) + tf_models.safe_log(1. - tf.nn.sigmoid(prior_discriminator - lg_p_z_prior)))

	# Eq (3.9)
	# NOTE: Take negative since we are minimizing
	elbo_loss = -tf.reduce_mean(-(discriminator - lg_p_z) + lg_p_x_given_z)

	iw_size = int(lg_p_x_given_z_iw.shape[0])
	iwae_loss = -tf.reduce_mean(tf.reduce_logsumexp(lg_p_x_given_z_iw - (discriminator_iw - lg_p_z_iw), axis=0) - tf.log(tf.constant(iw_size, dtype=tf.float32)))

	return tf.identity(elbo_loss, name=name+'/elbo_like'), tf.identity(discriminator_loss, name=name+'/discriminator'), tf.identity(iwae_loss, name=name+'/iwae')

def create(name='train', settings=None):
	lg_p_x_given_z = tf.squeeze(tf_models.get_output(name + '/p_x_given_z/log_prob'))
	discriminator = tf.squeeze(tf_models.get_output(name + '/discriminator/generator'))
	prior_discriminator = tf.squeeze(tf_models.get_output(name + '/discriminator/prior'))

	lg_p_x_given_z_iw = tf.squeeze(tf_models.get_output(name + '/p_x_given_z_iw/log_prob'))
	discriminator_iw = tf.squeeze(tf_models.get_output(name + '/discriminator/generator_iw'))
	#prior_discriminator_iw = tf.squeeze(tf_models.get_output(name + '/discriminator/prior_iw'))

	lg_p_z = tf.squeeze(tf_models.get_output(name + '/p_z/log_prob'))
	lg_p_z_iw = tf.squeeze(tf_models.get_output(name + '/p_z_iw/log_prob'))

	lg_p_z_prior = tf.squeeze(tf_models.get_output(name + '/p_z/log_prob_prior'))
	lg_p_z_iw_prior = tf.squeeze(tf_models.get_output(name + '/p_z_iw/log_prob_prior'))

	#print('lg_p_x_given_z_iw.shape', lg_p_x_given_z_iw.shape)
	#print('discriminator_iw.shape', discriminator_iw.shape)
	#print('prior_discriminator_iw.shape', prior_discriminator_iw.shape)
	#raise Exception()

	return loss(lg_p_x_given_z, discriminator, prior_discriminator, lg_p_x_given_z_iw, discriminator_iw, lg_p_z, lg_p_z_iw, lg_p_z_prior, lg_p_z_iw_prior, name=name)
