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

def create(settings):
	optimizer_lib = importlib.import_module('tensorflow_models.optimizers.' + settings['optimizer'])
	train_elbo_loss = tf_models.get_loss('train/elbo')
	step = tf_models.global_step()

	if not settings['optimizer'] is 'adam':
		train_op = optimizer_lib.training(train_elbo_loss, learning_rate=settings['learning_rate'], step=step, name='elbo')
	else:
		train_op = optimizer_lib.training(train_elbo_loss, learning_rate=settings['learning_rate'], step=step, name='elbo', beta1=settings['adam_beta1'], beta2=settings['adam_beta2'])
	return train_op