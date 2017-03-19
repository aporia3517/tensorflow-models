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

# Initializes TensorFlow and loads data
class Context(object):
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

class SessionContext(object):
	def __init__(self, settings, model):
		#super(SessionContext, self).__init__()
		self._settings = settings
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
			self.results = {}
			self.results['costs_train'] = []
			self.results['times_train'] = []
			self.results['costs_test'] = []
			if self._settings['inference'] == 'avb' or self._settings['inference'] == 'em-avb':
				self.results['adversarial_train'] = []

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

#class LearningContext(object):
#	def __init__(self, settings, model):