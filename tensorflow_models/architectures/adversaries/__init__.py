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
from six.moves import range

# Discriminator for GAN and critic for W-GANS (GAN code takes sigmoid of this!)
def adversary_mlp(settings, inputs, is_training):
	architecture = settings['architecture']
	params = architecture['adversary']
	return tf_models.layers.mlp(
					inputs,
					params['sizes'] + [1],
					activation_fn=params['activation_fn'],
					final_activation_fn=params['output_fn'])

# Discriminator used for adversarial training in logits
# Transforms each input separately before combining 
def split_2v_mlp(settings, x, z, is_training):
	architecture = settings['architecture']
	params = architecture['adversary']
	x = tf_models.flatten(x)

	x_layer = tf_models.layers.mlp(x, params['x_sizes'], scope='x_layer')
	z_layer = tf_models.layers.mlp(z, params['z_sizes'], scope='z_layer')
	return tf_models.layers.mlp(
						tf.concat([x_layer, z_layer], axis=1),
						params['join_sizes'] + [1], scope='join_layer',
						activation_fn=params['activation_fn'],
						final_activation_fn=params['output_fn'])

def adversary_2v_mlp(settings, x, z, is_training):
	x = tf_models.flatten(x)
	architecture = settings['architecture']
	params = architecture['adversary']
	inputs = tf.concat([x, z], axis=1)
	return tf_models.layers.mlp(
					inputs,
					params['sizes'] + [1],
					activation_fn=params['activation_fn'],
					final_activation_fn=params['output_fn'])

# DC-GAN discriminator
def adversary_dcgan(settings, inputs, is_training):
	architecture = settings['architecture']
	params = architecture['adversary']
	assert len(inputs.shape) == 4

	# Extract params and set defaults
	if 'batch_norm' in params and params['batch_norm']:
		normalizer_fn = slim.batch_norm
		normalizer_params = {'scale':True, 'is_training':is_training, 'updates_collections':None, 'decay':0.95}
	else:
		normalizer_fn = None
		normalizer_params = None

	activation_fn = params['activation_fn']
	output_fn = params['output_fn']

	#h = tf.reshape(inputs, [100, 28, 28, 1])
	h = inputs

	#print('DC-GAN discriminator')
	#print('h.shape', h.shape)

	df_dim = params['filter_count']
	initial_size = params['initial_size']

	dims = list(df_dim*(params['increase_factor']**np.arange(params['conv_layers'])))
	h = slim.conv2d(h, dims[0], activation_fn=activation_fn, normalizer_fn=None, normalizer_params=None, kernel_size=[5, 5], stride=2, padding='SAME', scope='convs_first')

	with slim.arg_scope([slim.conv2d],
                      activation_fn=activation_fn,
                      normalizer_fn=normalizer_fn,
											normalizer_params=normalizer_params,
											kernel_size=[5, 5],
											stride=2,
											padding='SAME'
											):
		
		#print('dims', dims)

		h = slim.stack(h, slim.conv2d, dims[1:], scope='convs')
		#print('h.shape', h.shape)

		h = tf.reshape(h, [100, -1])
		#print('h.shape', h.shape)

	h = slim.fully_connected(h, 1, activation_fn=output_fn, scope='output', normalizer_fn=None, normalizer_params=None)
	#print('h.shape', h.shape)

	return h
