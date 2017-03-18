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
import numpy as np

import tensorflow_models as tf_models

# Initializes TensorFlow, and loads data
class Context:
	def __init__(self, settings):
		self._graph = tf.Graph().as_default()
		self._settings = settings

	def __enter__(self):
		# Use the default graph
		self._graph.__enter__()

		# Fix seeds for reproducibility
		self._set_seeds()

		# Create input nodes and global step
		self._create_inputs()

		# Create some misc. variables
		self.train_batches, self.test_batches = tf_models.count_batches(self._settings)
		self.saver = tf.train.Saver()		
		
		return self

	def _set_seeds(self):
		np.random.seed(self._settings['np_random_seed'])
		tf.set_random_seed(self._settings['tf_random_seed'])

	def _create_inputs(self):
		with tf_models.cpu_device(self._settings):
			self.train_samples, self.test_samples = tf_models.unsupervised_inputs(self._settings)
			self.global_step = tf_models.global_step()

	def __exit__(self, *args):
		self._graph.__exit__(*args)

# Handles running queue runners using coordinator object
class CoordinatorContext:
	def __init__(self, settings, sess):
		self._settings = settings
		
		# Start input enqueue threads.
		self._coord = tf.train.Coordinator()
		self._exception_context = self._coord.stop_on_exception()
		self._threads = tf.train.start_queue_runners(sess=sess, coord=self._coord)
		

	def __enter__(self):
		# Terminate threads on an exception
		self._exception_context.__enter__()
		
		return self

	def __exit__(self, *args):
		self._coord.request_stop()
		self._coord.join(self._threads)
		self._exception_context.__exit__(*args)

	def running(self):
		return not self._coord.should_stop()

	def stop(self):
		self._coord.request_stop()