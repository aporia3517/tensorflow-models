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

import tensorflow as tf
import tensorflow_models as tf_models

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
	return tf_models.layers.mlp(inputs, architecture['discriminator_sizes'] + [1], final_activation_fn=tf.nn.sigmoid)

# Black-box encoder: q(z | x, eps)
# Returns a sample from z given x and epsilon
def real_generator_mlp(settings, inputs, eps, is_training):
	architecture = settings['architecture']
	return tf_models.layers.mlp(
						tf.concat([inputs, eps], axis=1), 
						architecture['generator_sizes'] + [settings['latent_dimension']],
						final_activation_fn=tf.identity)

# Discriminator used for adversarial training in logits
# Transforms each input separately before combining 
def split_2v_discriminator_mlp(settings, x, z, is_training):
	architecture = settings['architecture']

	x_layer = tf_models.layers.mlp(x, architecture['discriminator_x_sizes'], scope='x_layer')
	z_layer = tf_models.layers.mlp(z, architecture['discriminator_z_sizes'], scope='z_layer')
	return tf_models.layers.mlp(
						tf.concat([x_layer, z_layer], axis=1),
						architecture['discriminator_join_sizes'] + [1], scope='join_layer',
						final_activation_fn=tf.identity)

def gan_2v_discriminator_mlp(settings, x, z, is_training):
	architecture = settings['architecture']
	inputs = tf.concat([x, z], axis=1)
	return tf_models.layers.mlp(inputs, architecture['discriminator_sizes'] + [1], final_activation_fn=tf.nn.sigmoid)

def avb_2v_discriminator_mlp(settings, x, z, is_training):
	architecture = settings['architecture']
	inputs = tf.concat([x, z], axis=1)
	return tf_models.layers.mlp(inputs, architecture['discriminator_sizes'] + [1], final_activation_fn=tf.nn.identity)