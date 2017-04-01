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

# Initializes TensorFlow and loads data
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

# TODO: Make this inherit from ModelContext
class SessionContext(object):
	def __init__(self, settings, model):
		#super(SessionContext, self).__init__()
		self._settings = settings
		self.train_batches, self.test_batches = tf_models.count_batches(self._settings)	
		self._model = model
		self.saver = tf.train.Saver()	
		self._init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	def __enter__(self):
		# Start the Tensorflow session
		self.sess = tf.Session()
		self.sess.__enter__()

		# Initialize training variables to scratch or resume from previous step
		self._initialize()

		return self

	def __exit__(self, *args):
		self.sess.__exit__(*args)
		pass

	def _initialize(self):
		if self._settings['resume_from'] is None:
			self.sess.run(self._init_op)
			self.step = 0
			inference_lib = importlib.import_module('tensorflow_models.inference.' + self._settings['inference'])
			self.results = inference_lib.initialize_results()
			self.prior_noise = self._model.sample_prior()
		else:
			# Check that checkpoint file exists
			self.step = self._settings['resume_from']
			snapshot_filepath = tf_models.utils.settings.filepath(self._settings) + '-' + str(self.step)
			if not tf_models.utils.file.exists(snapshot_filepath + '.meta'):
				raise IOError('Snapshot at step {} does not exist'.format(self.step))
			self._saver.restore(self.sess, snapshot_filepath)
			self.results, self.prior_noise = tf_models.utils.snapshot.load_results(snapshot_filepath)
			print("Model restored from epoch {}".format(self._settings['resume_from']))

# Handles running queue runners using coordinator object
class CoordinatorContext(SessionContext):
	def __init__(self, settings, model):
		super(CoordinatorContext, self).__init__(settings, model)
		self._settings = settings
		
		# Input enqueue threads
		self._coord = tf.train.Coordinator()
		self._exception_context = self._coord.stop_on_exception()
		
	def __enter__(self):
		# Terminate threads on an exception
		super(CoordinatorContext, self).__enter__()
		self._threads = tf.train.start_queue_runners(sess=self.sess, coord=self._coord)
		self._exception_context.__enter__()
		return self

	def __exit__(self, *args):
		self._coord.request_stop()
		self._coord.join(self._threads)
		self._exception_context.__exit__(*args)
		super(CoordinatorContext, self).__exit__(*args)

	def running(self):
		return not self._coord.should_stop()

	def stop(self):
		self._coord.request_stop()

class LearningContext(CoordinatorContext):
	def __init__(self, settings, model, paths):
		super(LearningContext, self).__init__(settings, model)
		self._paths = paths
		self._initialize_counters()

	def __enter__(self):
		super(LearningContext, self).__enter__()
		self._set_learning_hooks()
		self._set_initialize_hook()
		self._set_step_hook()
		self._set_after_step_hook()
		self._initialize_hook(self)
		self._save_snapshot()
		return self

	def __exit__(self, *args):
		self._finished_hook()
		super(LearningContext, self).__exit__(*args)

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

	# Called before learning is started
	def _set_initialize_hook(self):
		inference_lib = importlib.import_module('tensorflow_models.inference.' + self._settings['inference'])
		self._initialize_hook = inference_lib.initialize_hook

	# Called every run()
	def _set_step_hook(self):
		inference_lib = importlib.import_module('tensorflow_models.inference.' + self._settings['inference'])
		self._step_hook = inference_lib.step_hook

	def _before_step_hook(self):
		pass

	def _set_after_step_hook(self):
		inference_lib = importlib.import_module('tensorflow_models.inference.' + self._settings['inference'])
		self._after_step_hook = inference_lib.after_step_hook

	# Convert step number into epoch number
	def epoch(self):
		return self.step * float(self._batches_per_step) / self.train_batches

	def running(self):
		return self.step < self._count_steps and super(LearningContext, self).running()

	def run(self):
		self._before_step_hook()
		self._step_hook(self)
		self.step += 1
		self._after_step_hook(self)
		self._save_snapshot()

	# Save snapshot of model parameters, results so far and the prior noise for plotting samples
	def _save_snapshot(self):
		if (not self._settings['steps_per_snapshot'] is None) and (self.step % self._settings['steps_per_snapshot'] == 0):
			self.saver.save(self.sess, tf_models.settings.snapshots_filepath(self._settings, self._paths), global_step=self.step)
			tf_models.snapshot.save_results(tf_models.settings.snapshots_filepath(self._settings, self._paths), self.step, self.results, self.prior_noise)
			if self._settings['plot_samples']:
				decoded_samples = self.sess.run(self._model.decoder, feed_dict={self._model.z: self.prior_noise})
				tf_models.plot.sample_grid(tf_models.settings.plots_filepath(self._settings, self._paths) + '-samples-' + str(self.step), decoded_samples.reshape(tf_models.unflattened_batchshape(self._settings)))

	def _set_learning_hooks(self):
		inference_lib = importlib.import_module('tensorflow_models.inference.' + self._settings['inference'])
		self.train, self.test = inference_lib.learning_hooks(self)

	def _finished_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))