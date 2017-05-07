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

import os, sys

import tensorflow as tf
import tensorflow_models as tf_models
from tensorflow_models.trainers import BaseTrainer

class Trainer(BaseTrainer):
	def finalize_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))

	def learning_hooks(self):
		return None, None

	def initialize_hook(self):
		# See where the test loss starts
		if self._settings['resume_from'] is None:
			raise Exception('AIS calculation must resume from a snapshot')
		else:
			test_loss = self.results['costs_test'][-1]
		print('epoch {:.3f}, test loss = {:.2f}'.format(self.epoch(), test_loss))

	def step_hook(self):
		pass

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		pass

	def initialize_results_hook(self):
		results = {}
		return results
