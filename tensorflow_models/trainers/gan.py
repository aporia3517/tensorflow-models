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
import importlib
from enum import Enum

import tensorflow as tf
import numpy as np

import tensorflow_datasets as tf_data

# Create the functions that perform learning and evaluation
def learning_hooks(session):
	sess = session.sess
	model = session._model
	test_batches = session.test_batches

	generator_train_op = model.train_ops['generator_loss']
	train_generator_loss_op = model.loss_ops['train_generator_loss']
	test_generator_loss_op = model.loss_ops['test_generator_loss']

	adversary_train_op = model.train_ops['adversary_loss']
	train_adversary_loss_op = model.loss_ops['train_adversary_loss']
	test_adversary_loss_op = model.loss_ops['test_adversary_loss']

	def train(count_steps):
			total_generator = 0.
			total_adversary = 0.
			for idx in range(count_steps):
				#_, _, this_adversary, this_generator = sess.run([generator_train_op, adversary_train_op, train_adversary_loss_op, train_generator_loss_op])

				# Try interweaving
				_, this_generator = sess.run([generator_train_op, train_generator_loss_op])
				#for jdx in range(3):
				_, this_adversary = sess.run([adversary_train_op, train_adversary_loss_op])

				total_adversary += this_adversary
				total_generator += this_generator
			return total_adversary / count_steps, total_generator / count_steps

	def test():
		total_generator = 0.
		total_adversary = 0.
		for idx in range(test_batches):
			this_adversary, this_generator = sess.run([test_adversary_loss_op, test_generator_loss_op])
			total_generator += this_generator
			total_adversary += this_adversary
		return total_adversary / test_batches, total_generator / test_batches

	return train, test

def initialize_hook(session):
	# See where the test loss starts
	if session._settings['resume_from'] is None:
		# Do a test evaluation before any training happens
		discriminator_loss, generator_loss = session.test()
		session.results['generator_losses'] += [generator_loss]
		session.results['discriminator_losses'] += [discriminator_loss]
	else:
		generator_loss = session.results['generator_losses'][-1]
		discriminator_loss = session.results['discriminator_losses'][-1]

	print('epoch {:.3f}, gen loss = {:.2f}, discr loss = {:.2f}'.format(session.epoch(), generator_loss, discriminator_loss))

def step_hook(session):
	with tf_models.timer.Timer() as train_timer:
		_, _ = session.train(session._batches_per_step)

	adversary_loss, generator_loss = session.test()

	session.results['generator_losses'] += [generator_loss]
	session.results['discriminator_losses'] += [adversary_loss]
	session.results['train_times'] += [train_timer.interval]
	
def after_step_hook(session):
	generator_loss = session.results['generator_losses'][-1]
	discriminator_loss = session.results['discriminator_losses'][-1]
	train_time = session.results['train_times'][-1]

	examples_per_sec = session._settings['batch_size'] * session._batches_per_step / train_time
	sec_per_batch = train_time / session._batches_per_step

	print('epoch {:.3f}, gen loss = {:.2f}, discr loss = {:.2f} ({:.1f} examples/sec)'.format(session.epoch(), generator_loss, discriminator_loss, examples_per_sec))

def initialize_results():
	results = {}
	results['generator_losses'] = []
	results['discriminator_losses'] = []
	results['train_times'] = []
	return results