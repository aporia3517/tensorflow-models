﻿# MIT License
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
		generator_train_op = tf_models.get_inference('generator')
		train_generator_loss_op = tf_models.get_loss('train/generator')
		#test_generator_loss_op = tf_models.get_loss('test/generator')

		discriminator_train_op = tf_models.get_inference('discriminator')
		train_discriminator_loss_op = tf_models.get_loss('train/discriminator')
		#test_discriminator_loss_op = tf_models.get_loss('test/discriminator')

		discriminator_steps = self._settings['adversary_steps']

		def train(count_steps):
			total_generator = 0.
			total_discriminator = 0.
			for idx in range(count_steps):
				#_, _, this_discriminator, this_generator = sess.run([generator_train_op, discriminator_train_op, train_discriminator_loss_op, train_generator_loss_op])

				# Try interweaving
				_, this_generator = self.sess.run([generator_train_op, train_generator_loss_op])
				for jdx in range(discriminator_steps):
					_, this_discriminator = self.sess.run([discriminator_train_op, train_discriminator_loss_op])

				total_discriminator += this_discriminator
				total_generator += this_generator
			return total_discriminator / count_steps, total_generator / count_steps

		#def test():
		#	total_generator = 0.
		#	total_discriminator = 0.
		#	for idx in range(self.test_batches):
		#		this_discriminator, this_generator = self.sess.run([test_discriminator_loss_op, test_generator_loss_op])
		#		total_generator += this_generator
		#		total_discriminator += this_discriminator
		#	return total_discriminator / self.test_batches, total_generator / self.test_batches

		return train, None #, test

	def initialize_hook(self):
		# See where the test loss starts
		if self._settings['resume_from'] is None:
			# Do a test evaluation before any training happens
			#discriminator_loss, generator_loss = self.test()
			discriminator_loss = 0
			generator_loss = 0
			self.results['generator_losses'] += [generator_loss]
			self.results['discriminator_losses'] += [discriminator_loss]
		else:
			generator_loss = self.results['generator_losses'][-1]
			discriminator_loss = self.results['discriminator_losses'][-1]
		print('epoch {:.3f}, gen loss = {:.2f}, discr loss = {:.2f}'.format(self.epoch(), generator_loss, discriminator_loss))

	def step_hook(self):
		with tf_models.timer.Timer() as train_timer:
			#_, _ = self.train(self._batches_per_step)
			discriminator_loss, generator_loss = self.train(self._batches_per_step)

		#discriminator_loss, generator_loss = self.test()

		self.results['generator_losses'] += [generator_loss]
		self.results['discriminator_losses'] += [discriminator_loss]
		self.results['train_times'] += [train_timer.interval]

	def before_step_hook(self):
		pass

	def after_step_hook(self):
		generator_loss = self.results['generator_losses'][-1]
		discriminator_loss = self.results['discriminator_losses'][-1]
		train_time = self.results['train_times'][-1]

		examples_per_sec = self._settings['batch_size'] * self._batches_per_step / train_time
		sec_per_batch = train_time / self._batches_per_step

		print('epoch {:.3f}, gen loss = {:.2f}, discr loss = {:.2f} ({:.1f} examples/sec)'.format(self.epoch(), generator_loss, discriminator_loss, examples_per_sec))

	def initialize_results_hook(self):
		results = {}
		results['generator_losses'] = []
		results['discriminator_losses'] = []
		results['train_times'] = []
		return results