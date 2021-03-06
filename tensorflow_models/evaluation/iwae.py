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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import range

import tensorflow_models as tf_models
import tensorflow_datasets as tf_data
from tensorflow_models.evaluation import BaseEvaluator

class Evaluator(BaseEvaluator):
	# Initialize results, loading results/parameters from snapshot if requested
	def _resume(self):
		# Check that checkpoint file exists
		self.step = self._settings['iwae_start'] - self._settings['iwae_increment']
		#self._load_snapshot()

	def _end_step(self):
		return min(self._settings['iwae_end'], self._settings['count_epochs'])

	def running(self):
		return self.step < self._end_step()

	def run(self, train):
		if self.step + self._settings['iwae_increment'] < self._end_step():
			self.step += self._settings['iwae_increment']
		else:
			self.step = self._end_step()
		
		self._load_snapshot()
		self.step_hook(train)

	def _load_snapshot(self):
		# Check that checkpoint file exists
		snapshot_filepath = tf_models.settings.snapshots_filepath(self._settings, self._paths) + '-' + str(self.step)
		if not tf_data.utils.file.exists(snapshot_filepath + '.meta'):
			raise IOError('Snapshot at step {} does not exist'.format(self.step))
		self._saver.restore(self.sess, snapshot_filepath)
		self.results = tf_models.snapshot.load_results(snapshot_filepath)
		#print("Model restored from epoch {}".format(self.epoch()))

	def finalize_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))

	def initialize_hook(self):
		self.iwae_results = {}
		self.iwae_results['iwae'] = []
		self.iwae_results['iwae_epoch'] = []

	def step_hook(self, train):
		# See where the test loss starts
		if train:
			elbo = self.results['costs_train'][-1]
			if 'iwae_train' in self.results:
				iwae = self.results['iwaes_train'][-1]
			else:
				iwae = None
			test_iwae_op = tf_models.get_loss('test/iwae')
			x_batch = self._tensors[1]
		else:
			elbo = self.results['costs_test'][-1]
			if 'iwae_test' in self.results:
				iwae = self.results['iwaes_test'][-1]
			else:
				iwae = None
			test_iwae_op = tf_models.get_loss('test/iwae')
			x_batch = self._tensors[0]
		
		x_placeholder = tf_models.test_placeholder()

		# TODO: Operations
		with tf_models.timer.Timer() as iwae_timer:
			# TODO: Average over all of test/train set
			this_iwae = 0
			for idx in range(self._settings['iwae_samples']):
				this_iwae += self.sess.run(test_iwae_op, feed_dict={x_placeholder: x_batch()})
			this_iwae /= self._settings['iwae_samples']
	
		if train:
			if iwae is not None:
				print('epoch {:.3f}, train loss = {:.2f}/{:.2f}/{:.2f}, {:.1f} sec'.format(self.epoch(), elbo, iwae, this_iwae, iwae_timer.interval))
			else:
				print('epoch {:.3f}, train loss = {:.2f}/_/{:.2f}, {:.1f} sec'.format(self.epoch(), elbo, this_iwae, iwae_timer.interval))
		else:
			if iwae is not None:
				print('epoch {:.3f}, test loss = {:.2f}/{:.2f}/{:.2f}, {:.1f} sec'.format(self.epoch(), elbo, iwae, this_iwae, iwae_timer.interval))
			else:
				print('epoch {:.3f}, test loss = {:.2f}/_/{:.2f}, {:.1f} sec'.format(self.epoch(), elbo, this_iwae, iwae_timer.interval))

		self.iwae_results['iwae'].append(this_iwae)
		self.iwae_results['iwae_epoch'].append(self.epoch())

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		pass