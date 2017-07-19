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

#def loss(ll_data, ll_fake, ll_one_minus_fake, name):
def loss(discriminator_real, discriminator_fake, x_real, x_fake, name, bandwidth):


	grad = tf.gradients(discriminator_fake, [x_fake])
	grad_norm_fake = tf.reduce_sum(tf.square(grad[0]), axis=1)

	grad = tf.gradients(discriminator_real, [x_real])
	grad_norm_real = tf.reduce_sum(tf.square(grad[0]), axis=1)

	penalty = tf.square(1. - tf.nn.sigmoid(discriminator_real)) * grad_norm_real + tf.square(tf.nn.sigmoid(discriminator_fake)) * grad_norm_fake
	#penalty = tf.square(1. - tf.nn.sigmoid(discriminator_fake)) * grad_norm_fake + tf.square(tf.nn.sigmoid(discriminator_real)) * grad_norm_real

	#grad2 = tf.gradients(discriminator, [x])

	#print(discriminator_fake.shape)
	#print(discriminator_real.shape)
	#print(penalty.shape)
	#raise Exception()

	# Maximising this as adversary wants to disciminate better
	#discriminator_objective = tf.reduce_mean(ll_data + ll_one_minus_fake)
	discriminator_objective = tf.reduce_mean(tf_models.safe_log(tf.nn.sigmoid(discriminator_real)) + tf_models.safe_log(1 - tf.nn.sigmoid(discriminator_fake)))

	if bandwidth != 0.:
		discriminator_objective -= tf.reduce_mean(penalty) * bandwidth / 2.

	# Also, maximising this instead of minimizing mean(ll_one_minus_fake)
	generator_objective = tf.reduce_mean(tf_models.safe_log(tf.nn.sigmoid(discriminator_fake)))

	return tf.identity(-discriminator_objective, name=name+'/discriminator'), tf.identity(-generator_objective, name=name+'/generator')

def create(name='train', settings=None):
	#lg_p_real = tf_models.get_output(name + '/p_x/log_prob_real')
	#lg_p_fake = tf_models.get_output(name + '/p_x/log_prob_fake')
	#lg_p_one_minus_fake = tf_models.get_output(name + '/p_x/log_one_minus_prob_fake')

	#print('outputs', [op.name for op in tf.get_collection(tf_models.GraphKeys.OUTPUTS)])

	discriminator_real = tf.squeeze(tf_models.get_output(name + '/discriminator/real'))
	discriminator_fake = tf.squeeze(tf_models.get_output(name + '/discriminator/fake'))

	x_real = get_input(name)
	x_fake = tf_models.get_output(name + '/generator/x/fake')

	#return loss(lg_p_real, lg_p_fake, lg_p_one_minus_fake, name=name)
	return loss(discriminator_real, discriminator_fake, x_real, x_fake, name=name, bandwidth=settings['avb_bandwidth'])

def get_input(name):
	ops = tf.get_collection(tf_models.GraphKeys.INPUTS)
	for op in ops:
		if 'inputs/'+name+'/samples' in op.name:
			return op
	raise ValueError('No loss operation with substring "{}" exists'.format(name))
