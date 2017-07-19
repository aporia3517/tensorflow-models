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

import tensorflow as tf
import tensorflow_models as tf_models


# Create the training operations
def create(settings):
	optimizer_lib = importlib.import_module('tensorflow_models.optimizers.' + settings['optimizer'])
	train_elbo_loss = tf_models.get_loss('train/elbo_like')
	train_critic_loss = tf_models.get_loss('train/critic')
	train_discriminator_loss = tf_models.get_loss('train/discriminator')
	train_elbo_avb_loss = tf_models.get_loss('train/elbo_avb')

	step = tf_models.global_step()

	# Divide variables into those we optimize for the ELBO and those for the adversarial training
	elbo_vars = [var for var in tf.trainable_variables() if (not var.name.startswith('model/critic')) and (not var.name.startswith('model/discriminator'))]

	# TODO: Check this does not include batch norm variables
	critic_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/critic')]

	discriminator_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/discriminator')]

	#print('critic_vars\n', critic_vars)
	#print('elbo vars\n', elbo_vars)

	coef = settings['emvb_mixed_coef']

	# Add to the Graph operations that train the model.
	if not settings['optimizer'] is 'adam':
		elbo_train_op = optimizer_lib.training(coef * train_elbo_loss + (1 - coef) * train_elbo_avb_loss, learning_rate=settings['learning_rate'], var_list=elbo_vars, step=step, name='elbo_like')
		critic_train_op = optimizer_lib.training(train_critic_loss, learning_rate=settings['critic_rate'], var_list=critic_vars, name='critic')
		discriminator_train_op = optimizer_lib.training(train_discriminator_loss, learning_rate=settings['discriminator_rate'], var_list=discriminator_vars, name='discriminator')
		elbo_avb_train_op = optimizer_lib.training(train_elbo_avb_loss, learning_rate=settings['learning_rate'], var_list=elbo_vars, step=step, name='elbo_avb')
	else:
		elbo_train_op = optimizer_lib.training(coef * train_elbo_loss + (1 - coef) * train_elbo_avb_loss, learning_rate=settings['learning_rate'], var_list=elbo_vars, step=step, name='elbo_like', beta1=settings['adam_beta1'], beta2=settings['adam_beta2'])
		critic_train_op = optimizer_lib.training(train_critic_loss, learning_rate=settings['critic_rate'], var_list=critic_vars, name='critic', beta1=settings['adam_beta1'], beta2=settings['adam_beta2'])
		discriminator_train_op = optimizer_lib.training(train_discriminator_loss, learning_rate=settings['discriminator_rate'], var_list=discriminator_vars, name='discriminator', beta1=settings['adam_beta1'], beta2=settings['adam_beta2'])
		elbo_avb_train_op = optimizer_lib.training(train_elbo_avb_loss, learning_rate=settings['learning_rate'], var_list=elbo_vars, step=step, name='elbo_avb', beta1=settings['adam_beta1'], beta2=settings['adam_beta2'])

	return elbo_train_op, critic_train_op, discriminator_train_op, elbo_avb_train_op

