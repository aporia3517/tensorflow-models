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
from tensorflow_models.evaluation import BaseEvaluator

class Evaluator(BaseEvaluator):
	def finalize_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))

	def initialize_hook(self):
		self.iwae_results = {}
		self.iwae_results['iwae'] = []
		self.iwae_results['iwae_epoch'] = []

	def step_hook(self):
		# See where the test loss starts
		test_elbo = self.results['costs_test'][-1]
		test_iwae = self.results['iwaes_test'][-1]
		test_iwae_op = tf_models.get_loss('test/iwae')
		x_test = tf_models.test_placeholder()

		# TODO: Operations
		with tf_models.timer.Timer() as iwae_timer:
			# TODO: Average over all of test/train set
			this_iwae = self.sess.run(test_iwae_op, feed_dict={x_test: self._tensors})
	
		print('epoch {:.3f}, test loss = {:.2f}/{:.2f}/{:.2f}, {:.1f} sec'.format(self.epoch(), test_elbo, test_iwae, this_iwae, iwae_timer.interval))
		self.iwae_results['iwae'].append(this_iwae)
		self.iwae_results['iwae_epoch'].append(self.epoch())

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		pass