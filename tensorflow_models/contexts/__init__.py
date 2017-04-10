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

import importlib

import tensorflow as tf
import numpy as np

import tensorflow_models as tf_models
import tensorflow_models.timer
import tensorflow_models.snapshot
import tensorflow_models.plot

# Sets a new or existing graph as the default
# Sets the random number seeds for TF and NumPy
# Creates a global step variable on the host
# If specified, sets operations under the context to a given device
# Stores the graph reference
class GraphContext(object):
	def __init__(self, graph=None, np_seed=1234, tf_seed=1234, device=None):
		if not graph is None:
			# TODO: Assert that graph is of correct class, tf.Graph?
			self.graph = graph
		else:
			self.graph = tf.Graph()

		if not device is None:
			self._device_context = tf.device(device)
		else:
			self._device_context = None

		self._np_seed = np_seed
		self._tf_seed = tf_seed

	def __enter__(self):
		# Use the context's graph for all subsequent operatons
		self._graph_context = self.graph.as_default()
		self._graph_context.__enter__()

		# Fix seeds for reproducibility
		self._set_seeds()

		# Create global step
		with tf_models.host():
			tf_models.global_step()

		# If specified, use a certain device by default
		if not self._device_context is None:
			self._device_context.__enter__()
		
		return self

	def _set_seeds(self):
		np.random.seed(self._np_seed)
		tf.set_random_seed(self._tf_seed)

	def __exit__(self, *args):
		if not self._device_context is None:
			self._device_context.__exit__(*args)
		self._graph_context.__exit__(*args)

class SessionContext(object):
	def __init__(self):
		self.saver = tf.train.Saver()
		tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)
		self._init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self._coord = None

		# Detect if there are queue runners and create necessary data structures if so
		if not tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS) is []:
			self._coord = tf.train.Coordinator()
			self._exception_context = self._coord.stop_on_exception()
		else:
			self._is_running = True

	def __enter__(self):
		# Start the Tensorflow session
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.sess.__enter__()
		self.sess.run(self._init_op)

		if not self._coord is None:
			self._threads = tf.train.start_queue_runners(sess=self.sess, coord=self._coord)
			self._exception_context.__enter__()

		return self

	def __exit__(self, *args):
		if not self._coord is None:
			self._coord.request_stop()
			self._coord.join(self._threads)
			self._exception_context.__exit__(*args)
		self.sess.__exit__(*args)

	def running(self):
		if not self._coord is None:
			return not self._coord.should_stop()
		else:
			return self._is_running

	def stop(self):
		if not self._coord is None:
			self._coord.request_stop()
		else:
			self._is_running = False