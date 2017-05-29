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

import numpy as np
import tensorflow as tf
import tensorflow_models as tf_models
from tensorflow_models.trainers import BaseTrainer

def log_mean_exp(x, axis=None):
	m = np.max(x, axis=axis, keepdims=True)
	return m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))

class Trainer(BaseTrainer):
	def finalize_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))

	# Create the functions that perform learning and evaluation
	def learning_hooks(self):
		critic_steps = self._settings['adversary_steps']

		elbo_train_op = tf_models.get_inference('elbo_like')
		train_elbo_loss_op = tf_models.get_loss('train/elbo_like')

		critic_train_op = tf_models.get_inference('critic')
		train_critic_loss_op = tf_models.get_loss('train/critic')

		test_elbo_loss_op = tf_models.get_loss('test/elbo_like')
		test_critic_loss_op = tf_models.get_loss('test/critic')

		def train(count_steps):
			total_elbo = 0.
			total_critic = 0.
			for idx in range(count_steps):
				# NOTE: Should I do the learning of the critic first?

				# Try interweaving
				#_, this_elbo, _, this_adversarial = self.sess.run([elbo_train_op, elbo_loss_op, adversarial_train_op, adversarial_loss_op])
				_, this_elbo = self.sess.run([elbo_train_op, train_elbo_loss_op])
				for jdx in range(critic_steps):
					_, this_critic = self.sess.run([critic_train_op, train_critic_loss_op])
				total_elbo += this_elbo
				total_critic += this_critic
			return total_elbo / count_steps, total_critic / count_steps

		def test():
			total_elbo = 0.
			total_critic = 0.
			for idx in range(self.test_batches):
				this_critic, this_elbo = self.sess.run([test_critic_loss_op, test_elbo_loss_op])
				total_elbo += this_elbo
				total_critic += this_critic
			return total_critic / self.test_batches, total_elbo / self.test_batches

		return train, test

	def initialize_hook(self):
		num_weights = self._settings['adapt_weights']
		batch_size = self._settings['batch_size']
		self.weights = np.zeros((num_weights, batch_size), dtype=np.float32)

		train_nll_op = tf_models.get_loss('train/nll')
		train_regularizer_op = tf_models.get_loss('train/regularizer')
		em_scale_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="em_scale")[0]
		momentum = self._settings['adapt_momentum']

		# Calculate AIS
		def calculate_ais():
			with tf_models.timer.Timer() as ais_timer:
				for i in range(num_weights):
					with tf_models.timer.Timer() as weights_timer:
						z, w, s, a = self.sess.run(self._ais_ops)
					print('importance weights {}/{}, {:.1f} sec'.format(i+1, num_weights, weights_timer.interval))
					self.weights[i, :] = w.reshape((batch_size,))

			print('Calculating AIS took {:.1f} sec'.format(ais_timer.interval))
			avg_weights = log_mean_exp(self.weights, axis=0)
			avg_nll = -np.mean(avg_weights)
			return avg_nll

		# Create function that adjusts
		def adapt_scale():
			# 1. Break up the EMVB bound into the NLL and regularizer terms
			print('Adapting EM scale at epoch {}'.format(self.epoch()))
			train_nll, train_regularizer, em_scale_old = self.sess.run([train_nll_op, train_regularizer_op, em_scale_var])
			print('Training nll {}, regularizer {}, old em scale {}'.format(train_nll, train_regularizer, em_scale_old))

			# 2. Calculate the AIS
			ais = calculate_ais()

			# 3. Adapt the EM scale
			em_scale_new = (train_nll - ais) / train_regularizer * em_scale_old
			em_moving_avg = (1. - momentum) * em_scale_old + momentum * em_scale_new
			print('AIS {}, New em scale {}, Moving avg of scale {}'.format(ais, em_scale_new, em_moving_avg))
			assign_op = em_scale_var.assign(em_moving_avg)
			self.sess.run(assign_op)

			# DEBUG
			raise Exception()

		self.adapt_em_scale = adapt_scale

		# See where the test loss starts
		if self._settings['resume_from'] is None:
			# Do a test evaluation before any training happens
			test_critic, test_elbo = self.test()
			self.results['elbo_test'] += [test_elbo]
			self.results['critic_test'] += [test_critic]
		else:
			test_elbo = self.results['elbo_test'][-1]
			test_critic = self.results['critic_test'][-1]

		#print('*** DEBUG ***')
		#print(test_elbo.shape)
		#print(test_critic.shape)

		print('epoch {:.3f}, test elbo = {:.2f}, test disc. = {:.2f}'.format(self.epoch(), test_elbo, test_critic))

	def step_hook(self):
		# Do adaptive step if necessary
		if self.step >= self._settings['adapt_start'] and self.step <= self._settings['adapt_end'] and (self.step - self._settings['adapt_start']) % self._settings['adapt_increment'] == 0:
			self.adapt_em_scale()

		with tf_models.timer.Timer() as train_timer:
			train_elbo, train_critic = self.train(self._batches_per_step)

		test_critic, test_elbo = self.test()

		self.results['times_train'] += [train_timer.interval]
		self.results['elbo_train'] += [train_elbo]
		self.results['critic_train'] += [train_critic]
		self.results['elbo_test'] += [test_elbo]
		self.results['critic_test'] += [test_critic]

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		train_time = self.results['times_train'][-1]
		test_elbo = self.results['elbo_test'][-1]
		test_critic = self.results['critic_test'][-1]
		train_elbo = self.results['elbo_train'][-1]

		examples_per_sec = self._settings['batch_size'] * self._batches_per_step / train_time
		sec_per_batch = train_time / self._batches_per_step

		print('epoch {:.3f}, test elbo = {:.2f}, test disc. = {:.2f}, train loss = {:.2f} ({:.1f} examples/sec)'.format(self.epoch(), test_elbo, test_critic, train_elbo, examples_per_sec))

	def initialize_results_hook(self):
		results = {}
		results['elbo_train'] = []
		results['critic_train'] = []
		results['times_train'] = []
		results['elbo_test'] = []
		results['critic_test'] = []
		return results
