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

# Implicit models for decoders

# TODO: Replace unit_mlp with mlp, and control output with params['activation_fn']

# Generate a flattened output on (0,1) using an MLP
def mlp(settings, code, is_training):
	architecture = settings['architecture']
	params = architecture['decoder']
	return tf_models.layers.mlp(
					code,
					params['sizes'] + tf_models.flattened_shape(settings),
					activation_fn=params['activation_fn'],
					final_activation_fn=params['activation_fn'])

# DC-GAN decoder. Uses transposed (fractionally strided) convolutions
def dcgan(settings, code, is_training):
	architecture = settings['architecture']
	params = architecture['decoder']
	batchshape = tf_models.batchshape(settings)
	assert len(batchshape) == 4

	# Extract params and set defaults
	if 'batch_norm' in params and params['batch_norm']:
		normalizer_fn = slim.batch_norm
		normalizer_params = {'scale':True, 'is_training':is_training, 'updates_collections':None, 'decay':0.95}
	else:
		normalizer_fn = None
		normalizer_params = None

	activation_fn = params['activation_fn']
	output_fn = params['output_fn']

	gf_dim = params['filter_count']
	initial_size = params['initial_size']
	final_filter_count = gf_dim * params['increase_factor']**(params['conv_layers']-1)

	# TODO: Made the following automated based on initial_size, layer_count etc.
	with slim.arg_scope([slim.fully_connected],
                      activation_fn=activation_fn,
                      normalizer_fn=normalizer_fn,
											normalizer_params=normalizer_params,
											):
		#print('DC-GAN generator')
		h = slim.fully_connected(code, initial_size[0]*initial_size[1]*final_filter_count, scope='projection')
		h = tf.reshape(h, [-1, initial_size[0], initial_size[1], final_filter_count])
		#print('h.shape', h.shape)

	with slim.arg_scope([slim.conv2d_transpose],
                      activation_fn=activation_fn,
                      normalizer_fn=normalizer_fn,
											normalizer_params=normalizer_params,
											kernel_size=[5, 5],
											stride=2,
											padding='SAME'
											):
		dims = list(reversed(list(gf_dim*(params['increase_factor']**np.arange(params['conv_layers'] - 1)))))
		#print('dims', dims)
		
		h = slim.stack(h, slim.conv2d_transpose, dims, scope='deconvs')
		#print('h.shape', h.shape)
		#h = slim.conv2d_transpose(h, gf_dim*2, scope='g2')
		#h = slim.conv2d_transpose(h, gf_dim, scope='g3')

		h = slim.conv2d_transpose(h, batchshape[3], activation_fn=output_fn, normalizer_fn=None, normalizer_params=None, scope='final_deconv')
		#print('h.shape', h.shape)
		h = tf.reshape(h, batchshape)
		#print('h.shape', h.shape)
	return h

