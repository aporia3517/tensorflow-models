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
	TRAIN_INPUTS = 'train_inputs'
	TEST_INPUTS = 'test_inputs'
	VALIDATION_INPUTS = 'validation_inputs'
	TRAIN_OUTPUTS = 'train_outputs'
	TEST_OUTPUTS = 'test_outputs'
	TEST_OUTPUTS = 'validation_outputs'

# Gets the shape of the tensor holding an unflattened minibatch => (batch x channels x height x width)
def unflattened_batchshape(settings):
	return (settings['batch_size'],) + tf_data.sample_shape(settings['dataset'])

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

	#with device(settings):
	#	#model(settings)
	#	#losses(settings)
	#	#optimizers(settings)

# TODO: Would it be better to expand the settings dictionary when it is called and have named arguments?
def inputs(settings):
	with tf.name_scope('train'):
		train_samples = tf_data.inputs(
			name=settings['dataset'],
			subset=tf_data.Subset.TRAIN,
			return_labels=settings['labels'],
			batch_size=settings['batch_size'],
			num_threads=settings['num_threads'],
			transformations=settings['transformations'])

	if settings['labels']:
		tf.add_to_collection(tf_models.GraphKeys.TRAIN_INPUTS, train_samples[0])
		tf.add_to_collection(tf_models.GraphKeys.TRAIN_INPUTS, train_samples[1])
	else:
		tf.add_to_collection(tf_models.GraphKeys.TRAIN_INPUTS, train_samples)

	with tf.name_scope('train'):
		test_samples = tf_data.inputs(
			name=settings['dataset'],
			subset=tf_data.Subset.TEST,
			return_labels=settings['labels'],
			batch_size=settings['batch_size'],
			num_threads=settings['num_threads'],
			transformations=settings['transformations'])

	if settings['labels']:
		tf.add_to_collection(tf_models.GraphKeys.TEST_INPUTS, test_samples[0])
		tf.add_to_collection(tf_models.GraphKeys.TEST_INPUTS, test_samples[1])
	else:
		tf.add_to_collection(tf_models.GraphKeys.TEST_INPUTS, test_samples)

	return train_samples, test_samples

# Load a model
class Model(object):
	def __init__(self, settings, train_samples, test_samples, step=None):
		# Save arguments for later
		self._settings = settings
		self._train_samples = train_samples
		self._test_samples = test_samples
		self._step = step

		# Create the operations to evaluate model and calculate loss
		self._create_model()

		self.z = self._model_train.z_placeholder
		self.x = self._model_train.x_placeholder

		# NOTE: GAN models etc. won't have an encoder
		try:
			self._model_train.encoder
		except AttributeError:
			self.encoder = None
		else:
			self.encoder = self._model_train.encoder
		
		self.decoder = self._model_train.decoder

		# Create the loss operations
		self._create_losses()

		# Create the optimizer
		self._create_optimizer()

	def _create_model(self):
		#print('Loading: tensorflow_models.models.' + self._settings['model'])
		model_lib = importlib.import_module('tensorflow_models.models.' + self._settings['model'])
		Model = model_lib.Model

		with gpu_device(self._settings):
			with tf.variable_scope(self._settings['model']):
				self._model_train = Model(self._train_samples, self._settings)
				tf.get_variable_scope().reuse_variables()
				self._model_test = Model(self._test_samples, self._settings)

	def _create_losses(self):
		loss_lib = importlib.import_module('tensorflow_models.losses.' + self._settings['loss'])

		# Make this agnostic to the inference method!
		with gpu_device(self._settings):
			with tf.variable_scope(self._settings['model']):
				# Add to the Graph the loss calculation.
				self.loss_ops = loss_lib.make(self._model_train.inference(), self._model_test.inference())

	def _create_optimizer(self):
		inference_lib = importlib.import_module('tensorflow_models.inference.' + self._settings['inference'])

		with gpu_device(self._settings):
			with tf.variable_scope(self._settings['model']):
				# Add to the Graph operations that train the model.
				self.train_ops = inference_lib.make(self._settings, self.loss_ops, self._step)

	def sample_prior(self):
		return self._model_train.sample_prior()