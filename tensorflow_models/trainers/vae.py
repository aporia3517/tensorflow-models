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

import tensorflow as tf
import numpy as np

import tensorflow_datasets as tf_data

def learning_hooks(session):
	sess = session.sess
	model = session._model
	test_batches = session.test_batches

	train_op = model.train_ops['train_loss']
	train_loss_op = model.loss_ops['train_loss']
	test_loss_op = model.loss_ops['train_loss']
	
	def train(count_steps):
		total_elbo = 0.
		for idx in range(count_steps):
			_, this_elbo = sess.run([train_op, train_loss_op])
			total_elbo += this_elbo
		return total_elbo / count_steps

	def test():
		total_loss = 0.
		for idx in range(test_batches):
			this_loss = sess.run(test_loss_op)
			total_loss += this_loss
		return total_loss / test_batches

	return train, test

def initialize_hook(session):
	# See where the test loss starts
	if session._settings['resume_from'] is None:
		# Do a test evaluation before any training happens
		test_loss = session.test()
		session.results['costs_test'] += [test_loss]
	else:
		test_loss = session.results['costs_test'][-1]

	print('epoch {:.3f}, test loss = {:.2f}'.format(session.epoch(), test_loss))

def step_hook(session):
	with tf_models.timer.Timer() as train_timer:
		train_loss = session.train(session._batches_per_step)
	test_loss = session.test()

	session.results['times_train'] += [train_timer.interval]
	session.results['costs_train'] += [train_loss]
	session.results['costs_test'] += [test_loss]

def after_step_hook(session):
	train_time = session.results['times_train'][-1]
	train_loss = session.results['costs_train'][-1]
	test_loss = session.results['costs_test'][-1]

	examples_per_sec = session._settings['batch_size'] * session._batches_per_step / train_time
	sec_per_batch = train_time / session._batches_per_step

	print('epoch {:.3f}, train loss = {:.2f}, test loss = {:.2f} ({:.1f} examples/sec)'.format(session.epoch(), train_loss, test_loss, examples_per_sec))

def initialize_results():
	results = {}
	results['costs_train'] = []
	results['times_train'] = []
	results['costs_test'] = []
	return results