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

#import tensorflow_models.trainers.avb
#import tensorflow_models.trainers.vae
#import tensorflow_models.trainers.gan

import tensorflow as tf
import tensorflow_datasets as tf_data
import tensorflow_models as tf_models

class BaseTrainer(object):
	def __init__(self, settings, paths, context):
		self._settings = settings
		self._context = context
		self.sess = self._context.sess
		self._paths = paths
		self._saver = tf.get_collection(tf.GraphKeys.SAVERS)[0]
		self.train_batches, self.test_batches = tf_models.count_batches(self._settings)
		self._initialize_counters()
		self._decoder = tf_models.get_decoder()
		self._z_placeholder = tf_models.codes_placeholder()

		if not self._decoder is None:
			assert(settings['batch_size'] >= settings['sample_size'])
		
	def __enter__(self):
		self._resume()
		self.train, self.test = self.learning_hooks()
		self.initialize_hook()
		self._save_snapshot()
		return self

	def __exit__(self, *args):
		self.finalize_hook()

	# Initialize results, loading results/parameters from snapshot if requested
	def _resume(self):
		if self._settings['resume_from'] is None:
			self.step = 0
			self.results = self.initialize_results_hook()
			if not self._decoder is None:
				self.results['prior_noise'] = self.sess.run(tf_models.get_prior())
		else:
			# Check that checkpoint file exists
			# TODO: Test this! I think it will fail on utils.settings.filepath because it doesn't have paths passed in
			self.step = self._settings['resume_from']
			snapshot_filepath = tf_models.settings.snapshots_filepath(self._settings, self._paths) + '-' + str(self.step)
			if not tf_data.utils.file.exists(snapshot_filepath + '.meta'):
				raise IOError('Snapshot at step {} does not exist'.format(self.step))
			self._saver.restore(self.sess, snapshot_filepath)

			# TODO: Version that omits prior noise for supervised learning
			# TODO: Need to remove decoding samples in other parts too!
			self.results = tf_models.snapshot.load_results(snapshot_filepath)
			print("Model restored from epoch {}".format(self._settings['resume_from']))

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

	def run(self):
		while self.running():
			self.before_step_hook()
			self.step_hook()
			self.step += 1
			self.after_step_hook()
			self._save_snapshot()

	# Convert step number into epoch number
	def epoch(self):
		return self.step * float(self._batches_per_step) / self.train_batches

	def running(self):
		return self.step < self._count_steps and self._context.running()

	def _save_snapshot(self):
		#print('saving snapshot')
		if (not self._settings['steps_per_snapshot'] is None) and (self.step % self._settings['steps_per_snapshot'] == 0):
			self._saver.save(self.sess, tf_models.settings.snapshots_filepath(self._settings, self._paths), global_step=self.step)
			tf_models.snapshot.save_results(tf_models.settings.snapshots_filepath(self._settings, self._paths), self.step, self.results)

			if self._settings['plot_samples'] and not self._decoder is None:
				#print('plotting samples: {}'.format(tf_models.settings.samples_filepath(self._settings, self._paths) + '-samples-' + str(self.step)))
				decoded_samples = self.sess.run(self._decoder, feed_dict={self._z_placeholder: self.results['prior_noise']})
				tf_models.plot.sample_grid(
					tf_models.settings.samples_filepath(self._settings, self._paths) + '-samples-' + str(self.step),
					decoded_samples.reshape(tf_models.unflattened_batchshape(self._settings)),
					self._settings['sample_size'],
					imgrange=tf_models.sample_scale(self._settings))
				#print('using scale', tf_models.sample_scale(self._settings))

	# Abtract methods
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
	def learning_hooks(self):
		raise NotImplementedError('Learning hooks factory has not been implemented')
	def initialize_results_hook(self):
		raise NotImplementedError('Results initialization hook has not been implemented')
