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
import tensorflow_datasets as tf_data
import tensorflow_models as tf_models

class BaseEvaluator(object):
	def __init__(self, settings, paths, context, ops=None, tensors=None):
		self._settings = settings
		self._context = context
		self.sess = self._context.sess
		self._paths = paths
		self._saver = tf.get_collection(tf.GraphKeys.SAVERS)[0]
		self.train_batches, self.test_batches = tf_models.count_batches(self._settings)
		self._initialize_counters()
		self._decoder = tf_models.get_decoder()
		self._z_placeholder = tf_models.codes_placeholder()
		self._ops = ops
		self._tensors = tensors

		#if not self._decoder is None:
		#	assert(settings['batch_size'] >= settings['sample_size'])
		
	def __enter__(self):
		self._resume()
		self.initialize_hook()
		return self

	def __exit__(self, *args):
		self.finalize_hook()

	# Initialize results, loading results/parameters from snapshot if requested
	def _resume(self):
		# Check that checkpoint file exists
		self.step = self._settings['ais_start'] - self._settings['ais_increment']
		#self._load_snapshot()

	def _end_step(self):
		return min(self._settings['ais_end'], self._settings['count_epochs'])

	def running(self):
		return self.step < self._end_step()

	def run(self):
		if self.step + self._settings['ais_increment'] < self._end_step():
			self.step += self._settings['ais_increment']
		else:
			self.step = self._end_step()
		
		self._load_snapshot()
		self.step_hook()

	def _load_snapshot(self):
		# Check that checkpoint file exists
		snapshot_filepath = tf_models.settings.snapshots_filepath(self._settings, self._paths) + '-' + str(self.step)
		if not tf_data.utils.file.exists(snapshot_filepath + '.meta'):
			raise IOError('Snapshot at step {} does not exist'.format(self.step))
		self._saver.restore(self.sess, snapshot_filepath)
		self.results = tf_models.snapshot.load_results(snapshot_filepath)
		print("Model restored from epoch {}".format(self.epoch()))

	# Work out how many steps to do, and how many minibatches per step
	def _initialize_counters(self):
		if not self._settings['batches_per_step'] is None:
			self._batches_per_step = self._settings['batches_per_step']
		else:
			self._batches_per_step = self.train_batches
		if not self._settings['count_steps'] is None:
			self._count_steps = self._settings['count_steps']
		elif not self._settings['count_epochs'] is None:
			# self.batches_per_step => Batches per step
			# self.train_batches => Batches per epoch
			self._count_steps = self._settings['count_epochs'] * float(self.train_batches) / self._batches_per_step

	# Convert step number into epoch number
	def epoch(self):
		return self.step * float(self._batches_per_step) / self.train_batches

	def step_to_epoch(self, step):
		return step * float(self._batches_per_step) / self.train_batches

	def initialize_hook(self):
		raise NotImplementedError('Initialization hook has not been implemented')
	def before_step_hook(self):
		raise NotImplementedError('Before step hook has not been implemented')
	def step_hook(self):
		raise NotImplementedError('Step hook has not been implemented')
	def after_step_hook(self):
		raise NotImplementedError('After step hook has not been implemented')
	def finalize_hook(self):
		raise NotImplementedError('Finalization hook has not been implemented')
