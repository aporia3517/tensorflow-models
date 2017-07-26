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

class Trainer(BaseTrainer):
	def finalize_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))

	# Create the functions that perform learning and evaluation
	def learning_hooks(self):
		discriminator_steps = self._settings['adversary_steps']

		elbo_train_op = tf_models.get_inference('elbo_like')
		train_elbo_loss_op = tf_models.get_loss('train/elbo_like')

		discriminator_train_op = tf_models.get_inference('discriminator')
		train_discriminator_loss_op = tf_models.get_loss('train/discriminator')

		test_elbo_loss_op = tf_models.get_loss('test/elbo_like')
		test_discriminator_loss_op = tf_models.get_loss('test/discriminator')

		lg_p_x_given_z = tf.squeeze(tf_models.get_output('train/p_x_given_z/log_prob'))
		discriminator = tf.squeeze(tf_models.get_output('train/discriminator/generator'))

		lg_p_z = tf.squeeze(tf_models.get_output('train/p_z/log_prob'))
		lg_q_z_given_x = tf.squeeze(tf_models.get_output('train/q_z_given_x/log_prob'))

		x_train = tf_models.train_placeholder()
		x_test = tf_models.test_placeholder()
		next_train_batch = self._train_batch
		next_test_batch = self._test_batch

		def train(count_steps, debug=False):
			total_elbo = 0.
			total_discriminator = 0.
			for idx in range(count_steps):
				# Try interweaving
				#_, this_elbo, _, this_adversarial = self.sess.run([elbo_train_op, elbo_loss_op, adversarial_train_op, adversarial_loss_op])
				X_mb = next_train_batch()
				_, this_elbo, this_lg_p_x_given_z, this_discriminator, this_lg_p_z, this_lg_q_z_given_x = self.sess.run([elbo_train_op, train_elbo_loss_op, lg_p_x_given_z, discriminator, lg_p_z, lg_q_z_given_x], feed_dict={x_train: X_mb})

				# DEBUG
				if debug and idx == 0:
					true_discriminator = -(this_lg_p_z - this_lg_q_z_given_x)

					print('lg_p_x_given_z: mean {:.2f}, median {:.2f}, std {:.2f}, min {:.2f}, max {:.2f}'.format(np.mean(this_lg_p_x_given_z), np.median(this_lg_p_x_given_z), np.std(this_lg_p_x_given_z), np.amin(this_lg_p_x_given_z), np.amax(this_lg_p_x_given_z)))
					print('discriminator: mean {:.2f}, median {:.2f}, std {:.2f}, min {:.2f}, max {:.2f}'.format(np.mean(this_discriminator), np.median(this_discriminator), np.std(this_discriminator), np.amin(this_discriminator), np.amax(this_discriminator)))
					print('true disc.   : mean {:.2f}, median {:.2f}, std {:.2f}, min {:.2f}, max {:.2f}'.format(np.mean(true_discriminator), np.median(true_discriminator), np.std(true_discriminator), np.amin(true_discriminator), np.amax(true_discriminator)))
					print('')

				for jdx in range(discriminator_steps):
					#if jdx > 0:
					#	old_discriminator = this_discriminator
					X_mb = next_train_batch()
					_, this_discriminator = self.sess.run([discriminator_train_op, train_discriminator_loss_op], feed_dict={x_train: X_mb})
					
					#if jdx > 0:
					#	print('{:.4f} change in discriminator'.format((this_discriminator - old_discriminator)/old_discriminator))
				total_elbo += this_elbo
				total_discriminator += this_discriminator
			return total_elbo/count_steps, total_discriminator/count_steps

		def test():
			total_elbo = 0.
			total_discriminator = 0.
			for idx in range(self.test_batches):
				X_mb = next_test_batch()
				this_discriminator, this_elbo = self.sess.run([test_discriminator_loss_op, test_elbo_loss_op], feed_dict={x_test: X_mb})
				total_elbo += this_elbo
				total_discriminator += this_discriminator
			return total_discriminator/self.test_batches, total_elbo/self.test_batches

		return train, test

	def initialize_hook(self):
		# See where the test loss starts
		if self._settings['resume_from'] is None:
			# Do a test evaluation before any training happens
			test_discriminator, test_elbo = self.test()
			self.results['elbo_test'] += [test_elbo]
			self.results['discriminator_test'] += [test_discriminator]
		else:
			test_elbo = self.results['elbo_test'][-1]
			test_discriminator = self.results['discriminator_test'][-1]

		print('epoch {:.3f}, test elbo = {:.2f}, test disc. = {:.7f}'.format(self.epoch(), test_elbo, test_discriminator))

	def step_hook(self):
		#print(self.epoch())

		with tf_models.timer.Timer() as train_timer:
			#train_elbo, train_discriminator = self.train(self._batches_per_step, self.epoch() >= 20.)
			train_elbo, train_discriminator = self.train(self._batches_per_step, True)

		test_discriminator, test_elbo = self.test()

		self.results['times_train'] += [train_timer.interval]
		self.results['elbo_train'] += [train_elbo]
		self.results['discriminator_train'] += [train_discriminator]
		self.results['elbo_test'] += [test_elbo]
		self.results['discriminator_test'] += [test_discriminator]

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		train_time = self.results['times_train'][-1]
		test_elbo = self.results['elbo_test'][-1]
		test_discriminator = self.results['discriminator_test'][-1]
		train_elbo = self.results['elbo_train'][-1]

		examples_per_sec = self._settings['batch_size'] * self._batches_per_step / train_time
		sec_per_batch = train_time / self._batches_per_step

		print('epoch {:.3f}, test elbo = {:.2f}, test disc. = {:.7f}, train loss = {:.2f} ({:.1f} examples/sec)'.format(self.epoch(), test_elbo, test_discriminator, train_elbo, examples_per_sec))

	def initialize_results_hook(self):
		results = {}
		results['elbo_train'] = []
		results['discriminator_train'] = []
		results['times_train'] = []
		results['elbo_test'] = []
		results['discriminator_test'] = []
		return results
