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

import importlib

def learning_hooks(session):
	sess = session.sess
	model = session._model
	test_batches = session.test_batches

	elbo_train_op = model.train_ops['train_loss']
	elbo_loss_op = model.loss_ops['train_loss']

	adversarial_train_op = model.train_ops['adversarial_loss']
	adversarial_loss_op = model.loss_ops['adversarial_loss']

	test_loss_op = model.loss_ops['train_loss']

	def train_epoch(count_steps):
			total_elbo = 0.
			total_adversarial = 0.
			for idx in range(count_steps):
				# Try interweaving
				_, this_elbo, _, this_adversarial = sess.run([elbo_train_op, elbo_loss_op, adversarial_train_op, adversarial_loss_op])
				#_, this_elbo = sess.run([elbo_train_op, elbo_loss_op])
				#_, this_adversarial = sess.run([adversarial_train_op, adversarial_loss_op])
				total_elbo += this_elbo
				total_adversarial += this_adversarial
			return total_elbo / count_steps, total_adversarial / count_steps

	def test():
		total_loss = 0.
		for idx in range(test_batches):
			this_loss = sess.run(test_loss_op)
			total_loss += this_loss
		return total_loss / test_batches

	return train, test

def make(settings, loss_ops, step):
	optimizer_lib = importlib.import_module('tensorflow_models.optimizers.' + settings['optimizer'])

	# Divide variables into those we optimize for the ELBO and those for the adversarial training
	elbo_vars = [var for var in tf.trainable_variables() if not var.name.startswith('vae/discriminator')]
	adversarial_vars = [var for var in tf.trainable_variables() if var.name.startswith('vae/discriminator')]

	# Add to the Graph operations that train the model.
	elbo_train_op = optimizer_lib.training(loss_ops['train_loss'], learning_rate=settings['learning_rate'], var_list=elbo_vars, step=step)
	adversarial_train_op = optimizer_lib.training(loss_ops['adversarial_loss'], learning_rate=settings['adversary_rate'], var_list=adversarial_vars)

	return {'train_loss': elbo_train_op, 'adversarial_loss': adversarial_train_op}