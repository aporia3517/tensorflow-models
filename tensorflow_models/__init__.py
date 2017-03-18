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

def cpu_device(settings):
	return tf.device("/cpu:0")

def gpu_device(settings):
	return tf.device("/" + settings['device'])

# TODO: Would it be better to expand the settings dictionary when it is called and have named arguments?
# NOTE: Have flatten as a flag rather than a setting because I believe it will depend on the model type which input shape is required
def unsupervised_inputs(settings, flatten=True):
	train_samples = tf_data.inputs(
		name=settings['dataset'],
		subset=tf_data.Subset.TRAIN,
		return_labels=False,
		batch_size=settings['batch_size'],
		num_threads=settings['num_threads'],
		flatten=flatten,
		transformations=settings['transformations'])

	test_samples = tf_data.inputs(
		name=settings['dataset'],
		subset=tf_data.Subset.TEST,
		return_labels=False,
		batch_size=settings['batch_size'],
		num_threads=settings['num_threads'],
		flatten=flatten,
		transformations=settings['transformations'])

	return train_samples, test_samples

def supervised_inputs(settings, flatten=True):
	train_samples, train_labels = tf_data.inputs(
		name=settings['dataset'],
		subset=tf_data.Subset.TRAIN,
		return_labels=True,
		batch_size=settings['batch_size'],
		num_threads=settings['num_threads'],
		flatten=flatten,
		transformations=settings['transformations'])

	test_samples, test_labels = tf_data.inputs(
		name=settings['dataset'],
		subset=tf_data.Subset.TEST,
		return_labels=True,
		batch_size=settings['batch_size'],
		num_threads=settings['num_threads'],
		flatten=flatten,
		transformations=settings['transformations'])

	return train_samples, train_labels, test_samples, test_labels

# Load a model
class Model(object):
	def __init__(self, settings, train_samples, test_samples, step=None):
		# Save arguments for later
		self._settings = settings
		self._train_samples = train_samples
		self._test_samples = test_samples
		self._step = step

		# Create the operations to evaluate model and calculate loss
		_create_model()

		self.z = self._model_train.z_placeholder
		self.x = self._model_train.x_placeholder
		self.encoder = self._model_train.encoder
		self.decoder = self._model_train.decoder

		# Create the loss operations
		_create_losses()

		# Create the optimizer
		_create_optimizer()

	def _create_model(self):
		# TODO: Load correct modules for Model
		# *** CONTINUE FROM HERE 19/3/2017 ***
		with gpu_device(self._settings):
			with tf.variable_scope(self._settings['model']):
				self._model_train = Model(self._train_samples, self._settings)
				tf.get_variable_scope().reuse_variables()
				self._model_test = Model(self._test_samples, self._settings)

	def _create_losses(self):
		# TODO: Load correct modules for loss
		# *** CONTINUE FROM HERE 19/3/2017 ***
		with gpu_device(self._settings):
			with tf.variable_scope(self._settings['model']):
				# Add to the Graph the loss calculation.
				train_loss_op = loss.loss(*self._model_train.inference())
				test_loss_op = loss.loss(*self._model_test.inference())
				self.loss_ops = {'train_loss': train_loss_op, 'test_loss': test_loss_op}

	def _create_optimizer(self):
		# TODO: Load correct modules for loss
		# *** CONTINUE FROM HERE 19/3/2017 ***
		with gpu_device(self._settings):
			with tf.variable_scope(self._settings['model']):
				# Add to the Graph operations that train the model.
				train_op = optimizer.training(self.loss_ops['train_loss'], learning_rate=self._settings['learning_rate'], step=self._step)
				self.train_ops = {'train_loss': train_op}