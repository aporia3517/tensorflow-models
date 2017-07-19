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

import os, sys, time

import tensorflow as tf
import tensorflow_models as tf_models
import tensorflow_datasets as tf_data
from tensorflow_models.trainers import BaseTrainer

class Trainer(BaseTrainer):
	def finalize_hook(self):
		print('Done training for {} epochs'.format(self.epoch()))

	# Create the functions that perform learning and evaluation
	def learning_hooks(self):
		critic_steps = self._settings['critic_steps']
		#discriminator_steps = self._settings['discriminator_steps']
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

		x_train = tf_models.train_placeholder()
		x_test = tf_models.test_placeholder()
		next_train_batch = self._train_batch
		next_test_batch = self._test_batch

		def train(count_steps):
			total_elbo = 0.
			total_disc = 0.

			# Decide whether to do EMVB or AVB
			if start_avb is None or self.step < start_avb:
				train_op = elbo_train_op
				loss_op = train_elbo_loss_op
				adv_train_op = critic_train_op
				adv_loss_op = train_critic_loss_op
				#print('Doing EMVB')
			else:
				train_op = elbo_avb_train_op
				loss_op = train_elbo_avb_loss_op
				adv_train_op = discriminator_train_op
				adv_loss_op = train_discriminator_loss_op
				#print('Doing AVB')
			
			for idx in range(count_steps):
				X_mb = next_train_batch()
				_, this_elbo = self.sess.run([train_op, loss_op], feed_dict={x_train: X_mb})
				for jdx in range(critic_steps):
					X_mb = next_train_batch()
					_, this_disc = self.sess.run([adv_train_op, adv_loss_op], feed_dict={x_train: X_mb})
				total_elbo += this_elbo
				total_disc += this_disc
			
			return total_elbo / count_steps, total_disc / count_steps

		def test():
			total_elbo = 0.
			total_disc = 0.

			# Decide whether to do EMVB or AVB
			if start_avb is None or self.step < start_avb:
				loss_op = test_elbo_loss_op
				adv_loss_op = test_critic_loss_op
			else:
				loss_op = test_elbo_avb_loss_op
				adv_loss_op = test_discriminator_loss_op

			for idx in range(self.test_batches):
				X_mb = next_test_batch()
				this_disc, this_elbo = self.sess.run([adv_loss_op, loss_op], feed_dict={x_test: X_mb})

				total_elbo += this_elbo
				total_disc += this_disc

			return total_elbo/self.test_batches, total_disc/self.test_batches

		return train, test

	def initialize_hook(self):
		# See where the test loss starts
		if self._settings['resume_from'] is None:
			# Do a test evaluation before any training happens
			test_elbo, test_discriminator = self.test()
			self.results['elbo_test'] += [test_elbo]
			self.results['discriminator_test'] += [test_discriminator]
		else:
			test_elbo = self.results['elbo_test'][-1]
			test_discriminator = self.results['discriminator_test'][-1]

		print('epoch {:.3f}, test elbo = {:.2f}, test critic = {:.2f}'.format(self.epoch(), test_elbo, test_discriminator))

	def step_hook(self):
		with tf_models.timer.Timer() as train_timer:
			train_elbo, train_discriminator = self.train(self._batches_per_step)

		test_elbo, test_discriminator  = self.test()

		self.results['times_train'] += [train_timer.interval]
		self.results['elbo_train'] += [train_elbo]
		self.results['elbo_test'] += [test_elbo]

		self.results['discriminator_test'] += [test_discriminator]
		self.results['discriminator_train'] += [train_discriminator]

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		train_time = self.results['times_train'][-1]
		test_elbo = self.results['elbo_test'][-1]
		test_discriminator = self.results['discriminator_test'][-1]
		train_elbo = self.results['elbo_train'][-1]

		#test_discriminator = self.results['discriminator_test'][-1]

		examples_per_sec = self._settings['batch_size'] * self._batches_per_step / train_time
		sec_per_batch = train_time / self._batches_per_step

		print('epoch {:.3f}, test elbo = {:.2f}, test critic = {:.2f}, train elbo = {:.2f} ({:.1f} examples/sec)'.format(self.epoch(), test_elbo, test_discriminator, train_elbo, examples_per_sec))

	def initialize_results_hook(self):
		results = {}
		results['elbo_train'] = []
		results['times_train'] = []
		results['elbo_test'] = []

		results['discriminator_test'] = []
		results['discriminator_train'] = []

		return results
