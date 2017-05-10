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

# Encoder: q(z | x)
# Returns the parameters for the normal distribution on z given x
def gaussian_parameters_mlp(settings, inputs, is_training):
	architecture = settings['architecture']
	return tf_models.layers.gaussian_parameters_mlp(inputs, architecture['encoder_sizes'] + [settings['latent_dimension']])

# Decoder: p(x | z)
# Returns parameters for bernoulli distribution on x given z
def bernoulli_parameters_mlp(settings, code, is_training):
	architecture = settings['architecture']
	return tf_models.layers.bernoulli_parameters_mlp(code, architecture['decoder_sizes'] + tf_models.flattened_shape(settings))

# Generate a flattened output on (0,1) using an MLP
def unit_generator_mlp(settings, code, is_training):
	architecture = settings['architecture']
	return tf_models.layers.mlp(code, architecture['generator_sizes'] + tf_models.flattened_shape(settings), final_activation_fn=tf.nn.sigmoid)

# Discriminator for GAN in probs
def gan_discriminator_mlp(settings, inputs, is_training):
	architecture = settings['architecture']
	return tf_models.layers.mlp(inputs, architecture['discriminator_sizes'] + [1], final_activation_fn=tf.identity)

def wgan_critic_mlp(settings, inputs, is_training):
	architecture = settings['architecture']
	return tf_models.layers.mlp(inputs, architecture['critic_sizes'] + [1], final_activation_fn=tf.identity)

# Black-box encoder: q(z | x, eps)
# Returns a sample from z given x and epsilon
def real_generator_mlp(settings, inputs, eps, is_training):
	architecture = settings['architecture']
	inputs = tf_models.flatten(inputs)
	return tf_models.layers.mlp(
						tf.concat([inputs, eps], axis=1), 
						architecture['generator_sizes'] + [settings['latent_dimension']],
						final_activation_fn=tf.identity)

def bernoulli_parameters_dcgan(settings, code, is_training):
	architecture = settings['architecture']
	params = architecture['decoder_params']
	batchshape = tf_models.batchshape(settings)
	assert len(batchshape) == 4

	# Extract params and set defaults
	if 'batch_norm' in params and params['batch_norm']:
		normalizer_fn = slim.batch_norm
		normalizer_params = {'scale':True, 'is_training':is_training, 'updates_collections':None, 'decay':0.95}
	else:
		normalizer_fn = None
		normalizer_params = None

	if 'activation_fn' in params:
		activation_fn = params['activation_fn']
	else:
		activation_fn = tf.nn.relu

	if 'output_fn' in params:
		output_fn = params['output_fn']
	else:
		output_fn = tf.nn.sigmoid

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

		h = slim.conv2d_transpose(h, batchshape[3], activation_fn=tf.identity, normalizer_fn=None, normalizer_params=None, scope='final_deconv')
		#print('h.shape', h.shape)
		h = tf.reshape(h, batchshape)
		#print('h.shape', h.shape)
	return h

def gaussian_parameters_dcgan(settings, inputs, is_training):
	architecture = settings['architecture']
	params = architecture['encoder_params']
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

	if 'activation_fn' in params:
		activation_fn = params['activation_fn']
	else:
		activation_fn = tf.nn.relu

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
			
		h = tf.reshape(h, [100, -1])
		#print('h.shape', h.shape)

	mean_z = slim.fully_connected(h, settings['latent_dimension'], activation_fn=tf.identity, scope='mean', normalizer_fn=None, normalizer_params=None)
	log_sigma_sq_z = slim.fully_connected(h, settings['latent_dimension'], activation_fn=tf.identity, scope='log_sigma_sq', normalizer_fn=None, normalizer_params=None)
	diag_stdev_z = tf.sqrt(tf.exp(log_sigma_sq_z))
	return mean_z, diag_stdev_z

def avb_bb_encoder_dcgan(settings, inputs, eps, is_training):
	architecture = settings['architecture']
	params = architecture['encoder_params']
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

	if 'activation_fn' in params:
		activation_fn = params['activation_fn']
	else:
		activation_fn = tf.nn.relu

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
			h = h + tf.reshape(slim.fully_connected(eps, int(np.prod(h.shape[1:])), activation_fn=tf.identity, scope='h{}_inject'.format(i+1), normalizer_fn=None, normalizer_params=None), h.shape)
			#print('h.name', h.name,'h.shape', h.shape)
			
		h = tf.reshape(h, [100, -1])
		#print('h.shape', h.shape)

	h = slim.fully_connected(h, settings['latent_dimension'], activation_fn=tf.identity, scope='output', normalizer_fn=None, normalizer_params=None)
	#print('h.shape', h.shape)

	#raise Exception()

	return h
	

# Discriminator used for adversarial training in logits
# Transforms each input separately before combining 
def split_2v_discriminator_mlp(settings, x, z, is_training):
	architecture = settings['architecture']
	x = tf_models.flatten(x)

	x_layer = tf_models.layers.mlp(x, architecture['discriminator_x_sizes'], scope='x_layer')
	z_layer = tf_models.layers.mlp(z, architecture['discriminator_z_sizes'], scope='z_layer')
	return tf_models.layers.mlp(
						tf.concat([x_layer, z_layer], axis=1),
						architecture['discriminator_join_sizes'] + [1], scope='join_layer',
						final_activation_fn=tf.identity)

def split_2v_critic_mlp(settings, x, z, is_training):
	x = tf_models.flatten(x)

	architecture = settings['architecture']
	x_layer = tf_models.layers.mlp(x, architecture['critic_x_sizes'], scope='x_layer')
	z_layer = tf_models.layers.mlp(z, architecture['critic_z_sizes'], scope='z_layer')
	return tf_models.layers.mlp(
						tf.concat([x_layer, z_layer], axis=1),
						architecture['critic_join_sizes'] + [1], scope='join_layer',
						final_activation_fn=tf.identity)

def gan_2v_discriminator_mlp(settings, x, z, is_training):
	x = tf_models.flatten(x)

	architecture = settings['architecture']
	inputs = tf.concat([x, z], axis=1)
	return tf_models.layers.mlp(inputs, architecture['discriminator_sizes'] + [1], final_activation_fn=tf.nn.sigmoid)

def avb_2v_discriminator_mlp(settings, x, z, is_training):
	x = tf_models.flatten(x)

	architecture = settings['architecture']
	inputs = tf.concat([x, z], axis=1)
	return tf_models.layers.mlp(inputs, architecture['discriminator_sizes'] + [1], final_activation_fn=tf.identity)

# TODO: Refactoring
def wgan_2v_critic_mlp(settings, x, z, is_training):
	x = tf_models.flatten(x)

	architecture = settings['architecture']
	inputs = tf.concat([x, z], axis=1)
	return tf_models.layers.mlp(inputs, architecture['critic_sizes'] + [1], final_activation_fn=tf.identity)

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

normalizer_params={'scale':True, 'is_training':True, 'updates_collections':None}

def dcgan_generator(settings, code, is_training):
	print('dcgan_generator is_training=', is_training)

	# TODO: DC-GAN implemenation
	gf_dim = 32
	h = slim.fully_connected(code, gf_dim*4*4*4, scope='projection', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params={'scale':True, 'decay':0.95, 'is_training':is_training, 'updates_collections':None})
	h = tf.reshape(h, [-1, 4, 4, gf_dim*4])
	#h = slim.conv2d_transpose(h, gf_dim*4, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu, scope='g2', normalizer_fn=slim.batch_norm, normalizer_params={'scale':True, 'is_training':is_training, 'updates_collections':None})
	h = slim.conv2d_transpose(h, gf_dim*2, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu, scope='g3', normalizer_fn=slim.batch_norm, normalizer_params={'scale':True, 'decay':0.95, 'is_training':is_training, 'updates_collections':None})
	h = slim.conv2d_transpose(h, gf_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu, scope='g4', normalizer_fn=slim.batch_norm, normalizer_params={'scale':True, 'decay':0.95, 'is_training':is_training, 'updates_collections':None})
	h = slim.conv2d_transpose(h, 1, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh, scope='g5')
	h = tf.reshape(h, tf_models.batchshape(settings))

	return h

def dcgan_discriminator(settings, inputs, is_training):
	print('dcgan_discriminator is_training=', is_training)

	# TODO: DC-GAN implementation
	#h = tf.reshape(inputs, [100, 28, 28, 1])
	h = inputs
	df_dim = 32
	h = slim.conv2d(h, df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu, scope='h1')
	h = slim.conv2d(h, 2*df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu, scope='h2', normalizer_fn=slim.batch_norm, normalizer_params={'scale':True, 'decay':0.95, 'is_training':is_training, 'updates_collections':None})
	h = slim.conv2d(h, 4*df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu, scope='h3', normalizer_fn=slim.batch_norm, normalizer_params={'scale':True, 'decay':0.95, 'is_training':is_training, 'updates_collections':None})
	#h = slim.conv2d(h, 8*df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu, scope='h4', normalizer_fn=slim.batch_norm, normalizer_params={'scale':True, 'is_training':is_training, 'updates_collections':None})
	h = tf.reshape(h, [100, -1])
	h = slim.fully_connected(h, 1, activation_fn=tf.identity, scope='h5')

	return h

def generator(settings, z, is_training):
	#normalizer_params = 
	gf_dim = 64
	h = slim.fully_connected(z, gf_dim*4*4*4, scope='projection', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params)
	h = tf.reshape(h, [-1, 4, 4, gf_dim*4])
	h = slim.conv2d_transpose(h, gf_dim*2, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu, scope='g2', normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params)
	#h = slim.conv2d_transpose(h, gf_dim*2, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu, scope='g3', normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params)
	h = slim.conv2d_transpose(h, gf_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu, scope='g4', normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params)
	h = slim.conv2d_transpose(h, 1, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh, scope='g5')
	h = tf.reshape(h, [100, 32, 32, 1])
	return h

def discriminator(settings, x, is_training):
	df_dim = 64
	h = slim.conv2d(x, df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu, scope='h1')
	h = slim.conv2d(h, 2*df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu, scope='h2', normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params)
	h = slim.conv2d(h, 4*df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu, scope='h3', normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params)
	#h = slim.conv2d(h, 8*df_dim, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu, scope='h4', normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params, weights_initializer=tf.truncated_normal_initializer(stddev=stddev), reuse=reuse)
	h = tf.reshape(h, [100, -1])
	h = slim.fully_connected(h, 1, activation_fn=tf.identity, scope='h5')
	return h

# DC-GAN decoder. Uses transposed (fractionally strided) convolutions
def gan_generator_cnn(settings, code, is_training):
	architecture = settings['architecture']
	params = architecture['generator_params']
	batchshape = tf_models.batchshape(settings)
	assert len(batchshape) == 4

	# Extract params and set defaults
	if 'batch_norm' in params and params['batch_norm']:
		normalizer_fn = slim.batch_norm
		normalizer_params = {'scale':True, 'is_training':is_training, 'updates_collections':None, 'decay':0.95}
	else:
		normalizer_fn = None
		normalizer_params = None

	if 'activation_fn' in params:
		activation_fn = params['activation_fn']
	else:
		activation_fn = tf.nn.relu

	if 'output_fn' in params:
		output_fn = params['output_fn']
	else:
		output_fn = tf.nn.sigmoid

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

# DC-GAN discriminator
def gan_discriminator_cnn(settings, inputs, is_training):
	architecture = settings['architecture']
	params = architecture['discriminator_params']
	assert len(inputs.shape) == 4

	# Extract params and set defaults
	if 'batch_norm' in params and params['batch_norm']:
		normalizer_fn = slim.batch_norm
		normalizer_params = {'scale':True, 'is_training':is_training, 'updates_collections':None, 'decay':0.95}
	else:
		normalizer_fn = None
		normalizer_params = None

	if 'activation_fn' in params:
		activation_fn = params['activation_fn']
	else:
		activation_fn = tf.nn.relu

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

	h = slim.fully_connected(h, 1, activation_fn=tf.identity, scope='output', normalizer_fn=None, normalizer_params=None)
	#print('h.shape', h.shape)

	return h