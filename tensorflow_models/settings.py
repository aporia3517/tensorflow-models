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

import os

# Create a dictionary of the default experiment settings
def defaults():
	return {
		'np_random_seed': 0, # Random seed for NumPy
		'tf_random_seed': 0, # Random seed for TensorFlow
		'device': 'gpu:0',
		'name': 'default',	# Name of experiment
		'suffix': None,	# Suffix to append to experiment name
		'dataset': 'mnist',	# Name of datasets passed to tf_data
		'num_threads': 2,	# Number of threads to use for queue runners
		'transformations': None,	# Transformations dictionary passed to tf_data
		'model': 'vae', # Name of model (for now, does nothing)
		'loss': 'elbo',
		'inference': 'svb',	# Type of inference: svb => Stochastic Variational Bayes, avb => Adversarial Variational Bayes
		'optimizer': 'adam',	# Module name for optimizer: ['adam', 'rmsprop', 'sgd']
		'learning_rate': 0.001,   # Learning rate
		'batch_size': 100,
		'count_epochs': 100, # How many epochs to train for
		#'count_steps': None,
		'count_steps_per_train': None,	# How many steps of learning to do per training epoch, defaults to 1 epoch's worth
		'plot_samples': False, # Whether to plot samples at snapshot epochs
		'epochs_per_snapshot': None, # Save model at epoch 0, 10, 20, etc., that can be loaded later
		'resume_from': None, # Resume from a snapshot. Set to epoch number
		#'clip_gradients': True,	# Whether to clip each individual gradient element
		#'normalize_gradients': True,	# Whether to normalize sum of gradients if it exceeds a threshold
		#'gradient_threshold': 1.0,	# Threshold for clipping each element of gradient
		#'gradient_max_norm': 5.0,	# Threshold for sum of gradient elements 
		
	}

def filename(settings):
	if settings['suffix'] is None:
		return settings['model'] + '-' + settings['dataset'] + '-' + settings['name']
	else:
		return settings['model'] + '-' + settings['dataset'] + '-' + settings['name'] + '-' + settings['suffix']

def snapshots_filepath(settings, paths):
	return os.path.join(paths['snapshots'], filename(settings))
	#return os.path.join(path, filename(settings))

def plots_filepath(settings, paths):
	return os.path.join(paths['plots'], filename(settings))
	#return os.path.join(path, filename(settings))

# Load the settings from a file that overwrites defaults
def load(filename = '', paths):
	# If no name, then return defaults
	if not filename:
		return defaults()

	# Check that settings path and file exist
	filename = os.path.join(paths['settings'], filename + '.py')
	if not os.path.exists(paths['settings']):
		raise NotADirectoryError('Settings path does not exist: {0}'.format(paths['settings']))

	if not os.path.exists(filename):
		raise IOError('Settings file does not exist: {0}'.format(filename))

	import imp
	exp = imp.load_source('settings', filename)
	return exp.settings