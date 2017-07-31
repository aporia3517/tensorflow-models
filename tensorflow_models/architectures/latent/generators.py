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

# TODO: Refactor the following so it calls another function with the activation function specified

# Black-box encoders: q(z | x, eps)
# Returns a sample from z given x and epsilon
def mlp(settings, inputs, eps, is_training):
	architecture = settings['architecture']
	params = architecture['encoder']
	inputs = tf_models.flatten(inputs)
	return tf_models.layers.mlp(
						tf.concat([inputs, eps], axis=1), 
						params['sizes'] + [settings['latent_dimension']],
						activation_fn=params['activation_fn'],
						final_activation_fn=params['output_fn'],
						normalizer_fn=params['normalizer_fn'])

def dcgan(settings, inputs, eps, is_training):
	architecture = settings['architecture']
	params = architecture['encoder']
	assert len(inputs.shape) == 4
	batchshape = tf_models.batchshape(settings)
	latentshape = tf_models.latentshape(settings)

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

	#size_image = int(np.prod(batchshape[1:]))
	#print('size of image', size_image, type(size_image))

	dims = list(df_dim*(params['increase_factor']**np.arange(params['conv_layers'])))
	h = slim.conv2d(h, dims[0], activation_fn=activation_fn, normalizer_fn=None, normalizer_params=None, kernel_size=[5, 5], stride=2, padding='SAME', scope='h0')
	#print('h.name', h.name,'h.shape', h.shape)

	h = h + tf.reshape(slim.fully_connected(eps, int(np.prod(h.shape[1:])), activation_fn=tf.identity, scope='h0_inject', normalizer_fn=None, normalizer_params=None), h.shape)
	#print('h.name', h.name,'h.shape', h.shape)

	with slim.arg_scope([slim.conv2d],
                      activation_fn=activation_fn,
                      normalizer_fn=normalizer_fn,
											normalizer_params=normalizer_params,
											kernel_size=[5, 5],
											stride=2,
											padding='SAME'
											):
		
		#print('dims', dims)
		for i in range(len(dims) - 1):
			h = slim.conv2d(h, dims[i + 1], scope='h{}'.format(i+1))
			#print('h.name', h.name,'h.shape', h.shape)

			if params['inject_noise']:
				h = h + tf.reshape(slim.fully_connected(eps, int(np.prod(h.shape[1:])), activation_fn=tf.identity, scope='h{}_inject'.format(i+1), normalizer_fn=None, normalizer_params=None), h.shape)
			#print('h.name', h.name,'h.shape', h.shape)
			
		h = tf.reshape(h, [inputs.shape.as_list()[0], -1])
		#print('h.shape', h.shape)

	h = slim.fully_connected(h, settings['latent_dimension'], activation_fn=output_fn, scope='output', normalizer_fn=None, normalizer_params=None)
	#print('h.shape', h.shape)

	#raise Exception()

	return h
