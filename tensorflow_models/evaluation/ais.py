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
from tensorflow_models.evaluation.hmc import hmc_step, hmc_updates
from tensorflow_models.evaluation import BaseEvaluator

class Evaluator(BaseEvaluator):
	def finalize_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))

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

def ais(sample_prior_op, lg_prior, lg_px_given_z, x, num_steps=20, count_intermediate=100):

	step_size = tf.Variable(tf.constant(0.1, shape=(x.shape[0],)), trainable=False)
	avg_acceptance_rate = tf.Variable(tf.constant(0.9, shape=(x.shape[0],)), trainable=False)

	def update(old_z, old_w, step_size, avg_acceptance_rate, i):
		# Construct the log-prior and scaled log-likelihood terms
		lg_likelihood = lambda z: lg_px_given_z(x, z) 
		scaled_lg_likelihood = lambda z: lg_likelihood(z) * (i + 1) / tf.constant(count_intermediate, dtype=tf.float32)
		intermediate_distribution = lambda z: lg_prior(z) + scaled_lg_likelihood(z)

		# Generate the sample z_i from z_{i-1} using an HMC transition
		new_z, accept_op = hmc_step(initial_pos=old_z, log_posterior=intermediate_distribution, step_size=step_size, num_steps=num_steps)
		new_step_size, new_avg_acceptance_rate = hmc_updates(accept_op, step_size, avg_acceptance_rate)

		# DEBUG
		#print('old_z.shape', new_z.shape)
		#print('new_z.shape', new_z.shape)
		#print('accept_op.shape', accept_op.shape)
		#print('new_step_size.shape', new_step_size.shape)
		#print('new_avg_acceptance_rate.shape', new_avg_acceptance_rate.shape)
		
		# Update the weights by adding 1/N lg p(x | z_{i-1})
		new_w = old_w + tf.reshape(lg_likelihood(old_z), (-1, 1)) / tf.constant(count_intermediate, dtype=tf.float32)

		#print('new_w', new_w.shape)

		return [new_z, new_w, new_step_size, new_avg_acceptance_rate, tf.add(i, 1)]
		#return [new_z, new_w, step_size, avg_acceptance_rate, tf.add(i, 1)]

	def condition(z, w, s, a, i):
		return tf.less(i, count_intermediate)

	# Generate z_0 from the prior, p(z)
	# TODO: Correct function to sample prior from model
	z = sample_prior_op

	# Initialize the weights to zero
	w = tf.zeros((x.shape[0], 1), dtype=tf.float32)
	i = tf.constant(0., dtype=tf.float32)

	# DEBUG
	#print('step_size.shape', step_size.shape)
	#print('avg_acceptance_rate.shape', avg_acceptance_rate.shape)
	#print('z.shape', z.shape)
	#print('w.shape', w.shape)

	final_z, final_w, new_step_size, new_acceptance_rate, _ = tf.while_loop(condition, update, [z, w, step_size, avg_acceptance_rate, i], parallel_iterations=1)

	return final_z, final_w, new_step_size, new_acceptance_rate
	#return final_z, final_w