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

import os, sys
import importlib
from enum import Enum

import tensorflow as tf
import numpy as np

import tensorflow_datasets as tf_data
import tensorflow_models.optimizers
import tensorflow_models.contexts
import tensorflow_models.layers

class GraphKeys(object):
	INPUTS = 'inputs'
	PRIOR = 'prior'
	PLACEHOLDERS = 'placeholders'
	OUTPUTS = 'outputs'
	ENCODERS = 'encoders'
	DECODERS = 'decoders'
	LOSSES = 'losses'
	OPTIMIZERS = 'optimizers'

# Gets the shape of the tensor holding an unflattened minibatch => (batch x channels x height x width)
def unflattened_batchshape(settings):
	return [settings['batch_size']] + list(tf_data.sample_shape(settings['dataset']))

def flattened_shape(settings):
	return [int(np.prod(tf_data.sample_shape(settings['dataset'])))]

def flattened_batchshape(settings):
	return [settings['batch_size']] + flattened_shape(settings)

def batchshape(settings):
	if 'flatten' in settings['transformations']:
		return flattened_batchshape(settings)
	else:
		return unflattened_batchshape(settings)

def safe_log(x, **kwargs):
	return tf.log(x + 1e-8, **kwargs)

def count_batches(settings, subset=None):
	if not subset is None:
		return tf_data.count(settings['dataset'], subset) // settings['batch_size']
	else:
		train_batches = count_batches(settings, tf_data.Subset.TRAIN)
		test_batches = count_batches(settings, tf_data.Subset.TEST)
		return train_batches, test_batches

def global_step():
	return tf.contrib.framework.get_or_create_global_step()

def local_step(settings, name='local_step', start=0):
	with cpu_device(settings):
		step = tf.Variable(start, name=name, trainable=False)

def host():
	return tf.device("/cpu:0")

def device(settings):
	return tf.device("/" + settings['device'])

def create(settings):
	with host():
		inputs(settings)

	with device(settings):
		model(settings)
		losses(settings)
		optimizers(settings)

# TODO: Would it be better to expand the settings dictionary when it is called and have named arguments?
def inputs(settings):
	with tf.name_scope('inputs/train'):
		train_samples = tf_data.inputs(
			name=settings['dataset'],
			subset=tf_data.Subset.TRAIN,
			return_labels=settings['labels'],
			batch_size=settings['batch_size'],
			num_threads=settings['num_threads'],
			transformations=settings['transformations'])

	for x in tf_data.utils.list.wrap(train_samples):
		tf.add_to_collection(GraphKeys.INPUTS, x)

	with tf.name_scope('inputs/test'):
		test_samples = tf_data.inputs(
			name=settings['dataset'],
			subset=tf_data.Subset.TEST,
			return_labels=settings['labels'],
			batch_size=settings['batch_size'],
			num_threads=settings['num_threads'],
			transformations=settings['transformations'])

	for x in tf_data.utils.list.wrap(test_samples):
		tf.add_to_collection(GraphKeys.INPUTS, x)

	return train_samples, test_samples

def samples(subset=tf_data.Subset.TRAIN):
	inputs = tf.get_collection(GraphKeys.INPUTS)
	for op in inputs:
		if tf_data.subset_suffix[subset] + '/samples' in op.name:
			return op
	return None

def labels(subset=tf_data.Subset.TRAIN):
	inputs = tf.get_collection(GraphKeys.INPUTS)
	for op in inputs:
		if tf_data.subset_suffix[subset] + '/labels' in op.name:
			return op
	return None

# Get trainable variables with a given substring
def vars(name):
	variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	selected = []
	for v in variables:
		if name in v.name:
			selected.append(v)
	return v

def outputs(name):
	ops = tf.get_collection(GraphKeys.OUTPUTS)
	for op in ops:
		if name in op.name:
			return op
	raise ValueError('No output operation with substring "{}" exists'.format(name))

def samples_placeholder():
	placeholders = tf.get_collection(GraphKeys.PLACEHOLDERS)
	for p in placeholders:
		if 'samples' in p.name:
			return p
	return None

def codes_placeholder():
	placeholders = tf.get_collection(GraphKeys.PLACEHOLDERS)
	for p in placeholders:
		if 'codes' in p.name:
			return p
	return None

def model(settings):
	model = importlib.import_module('tensorflow_models.models.' + settings['model'])
	
	with tf.variable_scope('model'):
		# Create and store an operation to sample from the prior
		if 'create_prior' in dir(model):
			with tf.name_scope('prior'):
				tf.add_to_collection(GraphKeys.PRIOR, model.create_prior(settings))

		with tf.name_scope('placeholders'):
			placeholders = model.create_placeholders(settings)
			for p in placeholders:
				tf.add_to_collection(GraphKeys.PLACEHOLDERS, p)

		with tf.name_scope('train'):
			probs = model.create_probs(settings, samples(tf_data.Subset.TRAIN))
			for p in probs:
				tf.add_to_collection(GraphKeys.OUTPUTS, p)

		with tf.name_scope('test'):
			probs = model.create_probs(settings, samples(tf_data.Subset.TEST), reuse=True)
			for p in probs:
				tf.add_to_collection(GraphKeys.OUTPUTS, p)

		tf.add_to_collection(GraphKeys.ENCODERS, model.create_encoder(settings, reuse=True))
		tf.add_to_collection(GraphKeys.DECODERS, model.create_decoder(settings, reuse=True))

def losses(settings):
	loss_lib = importlib.import_module('tensorflow_models.losses.' + settings['loss'])
	with tf.name_scope('losses'):
		tf.add_to_collection(GraphKeys.LOSSES, loss_lib.create('train'))
		tf.add_to_collection(GraphKeys.LOSSES, loss_lib.create('test'))

def optimizers(settings):
	pass

def latentshape(settings):
	return [settings['batch_size'], settings['latent_dimension']]

def standard_normal(shape, name='MultivariateNormalDiag'):
	return tf.contrib.distributions.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), name=name)

def standard_uniform(shape, name='Uniform'):
	return tf.contrib.distributions.Uniform(name=name)

def gan_uniform(shape, name='Uniform'):
	return tf.contrib.distributions.Uniform(a=-1., b=1., name=name)

#def _create_optimizer(self):
#	inference_lib = importlib.import_module('tensorflow_models.inference.' + self._settings['inference'])
#
#	with gpu_device(self._settings):
#		with tf.variable_scope(self._settings['model']):
#			# Add to the Graph operations that train the model.
#			self.train_ops = inference_lib.make(self._settings, self.loss_ops, self._step)