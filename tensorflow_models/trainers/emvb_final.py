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

	# Create the functions that perform learning and evaluation
	def learning_hooks(self):
		critic_steps = self._settings['critic_steps']
		discriminator_steps = self._settings['discriminator_steps']
		start_avb = self._settings['start_avb']

		elbo_train_op = tf_models.get_inference('elbo_like')
		train_elbo_loss_op = tf_models.get_loss('train/elbo_like')

		critic_train_op = tf_models.get_inference('critic')
		train_critic_loss_op = tf_models.get_loss('train/critic')

		discriminator_train_op = tf_models.get_inference('discriminator')
		train_discriminator_loss_op = tf_models.get_loss('train/discriminator')

		test_elbo_loss_op = tf_models.get_loss('test/elbo_like')
		test_critic_loss_op = tf_models.get_loss('test/critic')

		elbo_avb_train_op = tf_models.get_inference('elbo_avb')
		train_elbo_avb_loss_op = tf_models.get_loss('train/elbo_avb')
		test_elbo_avb_loss_op = tf_models.get_loss('test/elbo_avb')
		test_discriminator_loss_op = tf_models.get_loss('test/discriminator')

		def train(count_steps):
			total_elbo = 0.
			total_critic = 0.
			total_elbo_avb = 0.
			total_discriminator = 0.

			# Decide whether to do EMVB or AVB
			if start_avb is None or self.step < start_avb:
				train_op = elbo_train_op
				#print('Doing EMVB')
			else:
				train_op = elbo_avb_train_op
				#print('Doing AVB')

			if critic_steps == discriminator_steps:
				for idx in range(count_steps):
					_, this_elbo, this_elbo_avb = self.sess.run([train_op, train_elbo_loss_op, train_elbo_avb_loss_op])
					for jdx in range(critic_steps):
						_, this_critic, _, this_discriminator = self.sess.run([critic_train_op, train_critic_loss_op, discriminator_train_op, train_discriminator_loss_op])
					total_elbo += this_elbo
					total_critic += this_critic
					total_elbo_avb += this_elbo_avb
					total_discriminator += this_discriminator
			else:
				for idx in range(count_steps):
					_, this_elbo, this_elbo_avb = self.sess.run([train_op, train_elbo_loss_op, train_elbo_avb_loss_op])
					for jdx in range(critic_steps):
						_, this_critic = self.sess.run([critic_train_op, train_critic_loss_op])
					for jdx in range(discriminator_steps):
						_, this_discriminator = self.sess.run([discriminator_train_op, train_discriminator_loss_op])
					total_elbo += this_elbo
					total_critic += this_critic
					total_elbo_avb += this_elbo_avb
					total_discriminator += this_discriminator
			return total_elbo / count_steps, total_critic / count_steps, total_elbo_avb/count_steps, total_discriminator/count_steps

		def test():
			total_elbo = 0.
			total_critic = 0.
			total_elbo_avb = 0.
			total_discriminator = 0.

			for idx in range(self.test_batches):
				this_critic, this_elbo, this_discriminator, this_elbo_avb = self.sess.run([test_critic_loss_op, test_elbo_loss_op, test_discriminator_loss_op, test_elbo_avb_loss_op])

				total_elbo += this_elbo
				total_critic += this_critic
				total_elbo_avb += this_elbo_avb
				total_discriminator += this_discriminator

			return total_critic/self.test_batches, total_elbo/self.test_batches, total_discriminator/self.test_batches, total_elbo_avb/self.test_batches

		return train, test

	def initialize_hook(self):
		# See where the test loss starts
		if self._settings['resume_from'] is None:
			# Do a test evaluation before any training happens
			test_critic, test_elbo, test_discriminator, test_elbo_avb = self.test()
			self.results['elbo_test'] += [test_elbo]
			self.results['critic_test'] += [test_critic]
			self.results['elbo_avb_test'] += [test_elbo_avb]
			self.results['discriminator_test'] += [test_discriminator]
		else:
			test_elbo = self.results['elbo_test'][-1]
			test_critic = self.results['critic_test'][-1]
			test_elbo_avb = self.results['elbo_test_avb'][-1]
			test_discriminator = self.results['discriminator_test'][-1]

		#print('*** DEBUG ***')
		#print(test_elbo.shape)
		#print(test_critic.shape)

		print('epoch {:.3f}, test elbo = {:.2f}/{:.2f}, test critic = {:.2f}/{:.2f}'.format(self.epoch(), test_elbo, test_elbo_avb, test_critic, test_discriminator))

	def step_hook(self):
		with tf_models.timer.Timer() as train_timer:
			train_elbo, train_critic, train_elbo_avb, train_discriminator = self.train(self._batches_per_step)

		test_critic, test_elbo, test_discriminator, test_elbo_avb  = self.test()

		self.results['times_train'] += [train_timer.interval]
		self.results['elbo_train'] += [train_elbo]
		self.results['critic_train'] += [train_critic]
		self.results['elbo_test'] += [test_elbo]
		self.results['critic_test'] += [test_critic]

		self.results['elbo_avb_test'] += [test_elbo_avb]
		self.results['discriminator_test'] += [test_discriminator]
		self.results['elbo_avb_train'] += [train_elbo_avb]
		self.results['discriminator_train'] += [train_discriminator]

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		train_time = self.results['times_train'][-1]
		test_elbo = self.results['elbo_test'][-1]
		test_critic = self.results['critic_test'][-1]
		test_discriminator = self.results['discriminator_test'][-1]
		train_elbo = self.results['elbo_train'][-1]

		#test_discriminator = self.results['discriminator_test'][-1]

		train_elbo_avb = self.results['elbo_avb_train'][-1]
		test_elbo_avb = self.results['elbo_avb_test'][-1]

		examples_per_sec = self._settings['batch_size'] * self._batches_per_step / train_time
		sec_per_batch = train_time / self._batches_per_step

		print('epoch {:.3f}, test elbo = {:.2f}/{:.2f}, test critic = {:.2f}/{:.2f}, train elbo = {:.2f}/{:.2f} ({:.1f} examples/sec)'.format(self.epoch(), test_elbo, test_elbo_avb, test_critic, test_discriminator, train_elbo, train_elbo_avb, examples_per_sec))

	def initialize_results_hook(self):
		results = {}
		results['elbo_train'] = []
		results['critic_train'] = []
		results['times_train'] = []
		results['elbo_test'] = []
		results['critic_test'] = []

		results['elbo_avb_test'] = []
		results['discriminator_test'] = []
		results['elbo_avb_train'] = []
		results['discriminator_train'] = []

		return results
