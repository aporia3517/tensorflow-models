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
	train_generator_loss = tf_models.get_loss('train/generator')
	train_discriminator_loss = tf_models.get_loss('train/discriminator')
	step = tf_models.global_step()

	# Divide variables into those we optimize for the ELBO and those for the adversarial training
	generator_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/generator')]
	discriminator_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/discriminator')]
	#other_vars = [var for var in tf.trainable_variables() if not var.name.startswith(settings['model'] + '/generator') and not var.name.startswith(settings['model'] + '/discriminator')]

	# Use the optimizer to apply the gradients that minimize the loss
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	#print(update_ops)
	disc_update_ops = [v for v in update_ops if v.name.startswith('model/train/discriminator')]
	gen_update_ops = [v for v in update_ops if v.name.startswith('model/train/generator')]

	#print('')
	#print('disc_update_ops', disc_update_ops)
	#print('')
	#print('gen_update_ops', gen_update_ops)
	#print('')

	#print('generator variables')
	#print([var.name for var in generator_vars])
	#print('discriminator variables')
	#print([var.name for var in discriminator_vars])
	#print('other variables')

	# Add to the Graph operations that train the model.
	if not settings['optimizer'] is 'adam':
		generator_train_op = optimizer_lib.training(train_generator_loss, learning_rate=settings['learning_rate'], var_list=generator_vars, step=step, name='generator', update_ops=gen_update_ops)
		discriminator_train_op = optimizer_lib.training(train_discriminator_loss, learning_rate=settings['adversary_rate'], var_list=discriminator_vars, name='discriminator', update_ops=disc_update_ops)
	else:
		generator_train_op = optimizer_lib.training(train_generator_loss, learning_rate=settings['learning_rate'], var_list=generator_vars, step=step, name='generator', beta1=settings['adam_beta1'], beta2=settings['adam_beta2'])
		discriminator_train_op = optimizer_lib.training(train_discriminator_loss, learning_rate=settings['adversary_rate'], var_list=discriminator_vars, name='discriminator', beta1=settings['adam_beta1'], beta2=settings['adam_beta2'])

	return generator_train_op, discriminator_train_op
