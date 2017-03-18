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

from enum import Enum
import numpy as np

import os, sys
import importlib

import tensorflow as tf
import numpy as np

import tensorflow_data as tf_data

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

	return train_images, test_images

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

	return train_images, train_labels, test_images, test_labels