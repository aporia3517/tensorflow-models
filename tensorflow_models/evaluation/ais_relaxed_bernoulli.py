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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import range

import tensorflow_models as tf_models
from tensorflow_models.evaluation.hmc_relaxed_bernoulli import hmc_step, hmc_updates
from tensorflow_models.evaluation import BaseEvaluator

def log_mean_exp(x, axis=None):
	m = np.max(x, axis=axis, keepdims=True)
	return m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))

def tf_log_mean_exp(x, axis=None):
	m = tf.reduce_max(x, axis=axis, keep_dims=True)
	return m + tf.log(tf.reduce_mean(tf.exp(x - m), axis=axis, keep_dims=True))

class Evaluator(BaseEvaluator):
	def finalize_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))

	def initialize_hook(self):
		self.weights = np.zeros((self._settings['ais_num_weights'], self._settings['batch_size']), dtype=np.float32)
		self.ais_results = {}
		self.ais_results['ais_ll'] = []
		self.ais_results['ais_epoch'] = []

	def step_hook(self):
		# See where the test loss starts
		if 'costs_test' in self.results:
			test_loss = self.results['costs_test'][self.step]
		elif 'elbo_test' in self.results:
			test_loss = self.results['elbo_test'][-1]
		else:
			raise Exception("Can't find ELBO term in results")
		print('epoch {:.3f}, test loss = {:.2f}'.format(self.epoch(), test_loss))

		# TODO: Operations
		num_weights = self._settings['ais_num_weights']
		batch_size = self._settings['batch_size']
		with tf_models.timer.Timer() as ais_timer:
			for i in range(num_weights):
				with tf_models.timer.Timer() as weights_timer:
					z, eps, w, s, a = self.sess.run(self._ops)
				print('importance weights {}/{}, {:.1f} sec'.format(i+1, num_weights, weights_timer.interval))
				self.weights[i, :] = w.reshape((batch_size,))

		print('Calculating AIS took {:.1f} sec'.format(ais_timer.interval))
		avg_weights = log_mean_exp(self.weights, axis=0)
		avg_nll = -np.mean(avg_weights)
	
		print('estimate of mean(ll): ', avg_nll)
		self.ais_results['ais_ll'].append(avg_nll)
		self.ais_results['ais_epoch'].append(self.epoch())

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		pass	

def ais(sample_noise_op, sample_encoder_op_factory, lg_prior, lg_px_given_z, lg_qz_given_x_eps, lg_noise, x, num_steps=20, count_intermediate=100):

	step_size = tf.Variable(tf.constant(0.1, shape=(x.shape[0],)), trainable=False)
	avg_acceptance_rate = tf.Variable(tf.constant(0.9, shape=(x.shape[0],)), trainable=False)

	def update(old_z, old_eps, old_lg_w, step_size, avg_acceptance_rate, i):
		# Construct the log-prior and scaled log-likelihood terms
		lg_likelihood = lambda z: lg_px_given_z(x, z) 
		lg_encoder = lambda z, eps: lg_qz_given_x_eps(x, z, eps) 

		# i = 0,...,N-1 => i+1 = 1,...,N as desired
		beta = (i + 1.) / tf.constant(count_intermediate, dtype=tf.float32)
		intermediate_distribution = lambda z, eps: lg_noise(eps) + (1. - beta) * lg_encoder(z, eps) + beta * (lg_prior(z) + lg_likelihood(z))

		# Generate the sample z_i from z_{i-1} using an HMC transition
		# TODO: M
		new_z, new_eps, accept_op = hmc_step(initial_pos_z=old_z, initial_pos_eps=old_eps, log_posterior=intermediate_distribution, step_size=step_size, num_steps=num_steps)
		new_step_size, new_avg_acceptance_rate = hmc_updates(accept_op, step_size, avg_acceptance_rate)

		# DEBUG
		#print('old_z.shape', new_z.shape)
		#print('new_z.shape', new_z.shape)
		#print('accept_op.shape', accept_op.shape)
		#print('new_step_size.shape', new_step_size.shape)
		#print('new_avg_acceptance_rate.shape', new_avg_acceptance_rate.shape)
		
		# Update the weights in log-space
		lg_w_delta = (lg_likelihood(old_z) + lg_prior(old_z) - lg_encoder(old_z, old_eps)) / tf.constant(count_intermediate, dtype=tf.float32)
		new_lg_w = old_lg_w + tf.reshape(lg_w_delta, (-1, 1))

		#print('new_w', new_w.shape)

		return [new_z, new_eps, new_lg_w, new_step_size, new_avg_acceptance_rate, tf.add(i, 1)]
		#return [new_z, new_w, step_size, avg_acceptance_rate, tf.add(i, 1)]

	def condition(z, eps, w, s, a, i):
		return tf.less(i, count_intermediate)

	# Generate z_0 from the prior, p(z)
	# TODO: Correct function to sample prior from model

	eps = sample_noise_op
	z = sample_encoder_op_factory(x, eps)

	# Initialize the weights to zero
	w = tf.zeros((x.shape[0], 1), dtype=tf.float32)
	i = tf.constant(0., dtype=tf.float32)

	# DEBUG
	#print('step_size.shape', step_size.shape)
	#print('avg_acceptance_rate.shape', avg_acceptance_rate.shape)
	#print('z.shape', z.shape)
	#print('w.shape', w.shape)

	final_z, final_eps, final_w, new_step_size, new_acceptance_rate, _ = tf.while_loop(condition, update, [z, eps, w, step_size, avg_acceptance_rate, i], parallel_iterations=1)

	return final_z, final_eps, final_w, new_step_size, new_acceptance_rate
	#return final_z, final_w
