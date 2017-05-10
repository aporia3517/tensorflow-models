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

import os, sys, six
import importlib
from enum import Enum

import tensorflow as tf
import numpy as np

import tensorflow_datasets as tf_data
from tensorflow_datasets.utils.list import wrap
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
	INFERENCE = 'inference'

# Flatten out the HWC dimensions of a tensor
def flatten(x):
	if len(x.shape) > 2:
		x = tf.reshape(x, [int(x.shape[0]), -1])
	return x

# Sum out the HWC dimensions of a tensor
def reduce_sum(x):
	return tf.reduce_sum(flatten(x), 1)

# Return the scale of samples (which is [0, 1] unless transformations have been applied)
def sample_scale(settings):
	scale = [0, 1]
	if 'transformations' in settings:
		for k, v in six.viewitems(settings['transformations']):
			if k == 'rescale':
				scale = list(v)
	return scale

# Gets the shape of the tensor holding an unflattened minibatch => (batch x channels x height x width)
def unflattened_batchshape(settings):
	return [settings['batch_size']] + tf_data.unflattened_sample_shape(settings)

def flattened_shape(settings):
	return [int(np.prod(tf_data.unflattened_sample_shape(settings)))]

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

def create(settings, placeholders=False):
	with host():
		if not placeholders:
			input_ops(settings)
		else:
			input_placeholders(settings)

	with device(settings):
		model_ops(settings)
		loss_ops(settings)
		inference_ops(settings)

def trainer(settings):
	lib = importlib.import_module('tensorflow_models.trainers.' + settings['trainer'])
	return lib.Trainer

# TODO: Would it be better to expand the settings dictionary when it is called and have named arguments?
def input_ops(settings):
	with tf.name_scope('inputs/train'):
		train_samples = tf_data.inputs(
			name=settings['dataset'],
			subset=tf_data.Subset.TRAIN,
			return_labels=settings['labels'],
			batch_size=settings['batch_size'],
			num_threads=settings['num_threads'],
			transformations=settings['transformations'])

	for x in wrap(train_samples):
		tf.add_to_collection(GraphKeys.INPUTS, x)

	if not settings['model'] == 'gan':
		with tf.name_scope('inputs/test'):
			test_samples = tf_data.inputs(
				name=settings['dataset'],
				subset=tf_data.Subset.TEST,
				return_labels=settings['labels'],
				batch_size=settings['batch_size'],
				num_threads=settings['num_threads'],
				transformations=settings['transformations'])

		for x in wrap(test_samples):
			tf.add_to_collection(GraphKeys.INPUTS, x)

	#return train_samples, test_samples

def input_placeholders(settings):
	#count_train = settings['count'][tf_data.Subset.TRAIN]
	#count_test = settings['count'][tf_data.Subset.TEST]

	with tf.name_scope('inputs/train'):
		#print('sample_shape', sample_shape(settings), unflattened_sample_shape(settings))
		
		train = tf.placeholder(dtype=tf.float32, shape=np.concatenate(([None], tf_data.sample_shape(settings))), name='samples')
		if settings['labels']:
			train = [train, tf.placeholder(dtype=tf.float32, shape=[None, 1], name='labels')]

	for x in wrap(train):
		tf.add_to_collection(GraphKeys.INPUTS, x)

	with tf.name_scope('inputs/test'):
		test = tf.placeholder(dtype=tf.float32, shape=np.concatenate(([None], tf_data.sample_shape(settings))), name='samples')
		if settings['labels']:
			test = [test, tf.placeholder(dtype=tf.float32, shape=[None, 1], name='labels')]

	for x in wrap(test):
		tf.add_to_collection(GraphKeys.INPUTS, x)

	#print('train.shape', train.shape, 'test.shape', test.shape)
	#print('train', train, 'test', test)
	#raise Exception()

	return train, test

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
def get_vars(name):
	variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	selected = []
	for v in variables:
		if name in v.name:
			selected.append(v)
	return v

def get_prior():
	ops = tf.get_collection(GraphKeys.PRIOR)
	if not ops is []:
		return ops[0]
	else:
		raise ValueError('No prior sampling operation exists')

def lg_likelihood(settings):
	model = importlib.import_module('tensorflow_models.models.' + settings['model'])
	if 'lg_likelihood' in dir(model):
		return model.lg_likelihood
	else:
		raise ValueError('No log-likelihood function exists')

def lg_prior(settings):
	model = importlib.import_module('tensorflow_models.models.' + settings['model'])
	if 'lg_prior' in dir(model):
		return model.lg_prior
	else:
		raise ValueError('No log-prior function exists')

def get_decoder():
	ops = tf.get_collection(GraphKeys.DECODERS)
	if not ops is []:
		return ops[0]
	else:
		return None
		#raise ValueError('No decoder sampling operation exists')

def get_encoder():
	ops = tf.get_collection(GraphKeys.ENCODERS)
	if not ops is []:
		return ops[0]
	else:
		return None
		#raise ValueError('No encoder sampling operation exists')

def get_output(name):
	ops = tf.get_collection(GraphKeys.OUTPUTS)
	for op in ops:
		if name in op.name:
			return op
	raise ValueError('No output operation with substring "{}" exists'.format(name))

def get_loss(name):
	ops = tf.get_collection(GraphKeys.LOSSES)
	for op in ops:
		if name in op.name:
			return op
	raise ValueError('No loss operation with substring "{}" exists'.format(name))

def get_inference(name):
	ops = tf.get_collection(GraphKeys.INFERENCE)
	for op in ops:
		if name in op.name:
			return op
	raise ValueError('No inference operation with substring "{}" exists'.format(name))

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

def model_ops(settings):
	model = importlib.import_module('tensorflow_models.models.' + settings['model'])
	
	with tf.variable_scope('model'):
		# Create and store an operation to sample from the prior
		if 'create_prior' in dir(model):
			with tf.name_scope('prior'):
				tf.add_to_collection(GraphKeys.PRIOR, model.create_prior(settings))

		if 'create_placeholders' in dir(model):
			with tf.name_scope('placeholders'):
				placeholders = model.create_placeholders(settings)
				for p in wrap(placeholders):
					tf.add_to_collection(GraphKeys.PLACEHOLDERS, p)

		with tf.name_scope('train'):
			probs = model.create_probs(settings, samples(tf_data.Subset.TRAIN), is_training=True)
			for p in wrap(probs):
				tf.add_to_collection(GraphKeys.OUTPUTS, p)

		if not settings['model'] == 'gan':
			with tf.name_scope('test'):
				probs = model.create_probs(settings, samples(tf_data.Subset.TEST), is_training=False, reuse=True)
				for p in wrap(probs):
					tf.add_to_collection(GraphKeys.OUTPUTS, p)

		if 'create_encoder' in dir(model):
			tf.add_to_collection(GraphKeys.ENCODERS, model.create_encoder(settings, reuse=True))
		if 'create_decoder' in dir(model):
			tf.add_to_collection(GraphKeys.DECODERS, model.create_decoder(settings, reuse=True))

def loss_ops(settings):
	loss_lib = importlib.import_module('tensorflow_models.losses.' + settings['loss'])
	with tf.name_scope('losses'):
		if not settings['model'] == 'gan':
			ls = wrap(loss_lib.create('train')) + wrap(loss_lib.create('test'))
		else:
			ls = wrap(loss_lib.create('train'))
		for l in ls:
			tf.add_to_collection(GraphKeys.LOSSES, l)

def inference_ops(settings):
	inference_lib = importlib.import_module('tensorflow_models.inference.' + settings['inference'])
	with tf.name_scope('inference'):
		ops = wrap(inference_lib.create(settings))
		for op in ops:
			tf.add_to_collection(GraphKeys.INFERENCE, op)

def latentshape(settings):
	return [settings['batch_size'], settings['latent_dimension']]

def noiseshape(settings):
	return [settings['batch_size'], settings['noise_dimension']]

def standard_normal(shape, name='MultivariateNormalDiag'):
	return tf.contrib.distributions.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), name=name)

def standard_uniform(name='Uniform'):
	return tf.contrib.distributions.Uniform(name=name)

def gan_uniform(name='Uniform'):
	try:
		return tf.contrib.distributions.Uniform(a=-1., b=1., name=name)
	except:
		return tf.contrib.distributions.Uniform(low=-1., high=1., name=name)