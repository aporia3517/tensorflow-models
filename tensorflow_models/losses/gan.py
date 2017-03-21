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

import numpy as np
import tensorflow as tf

# The basic ELBO loss

# lg_p_x_given_z ~ batch_size x 784
# lg_p_z
# lg_q_z_given_x ~ batch_size?
def loss(ll_data, ll_fake, ll_one_minus_fake):
	# Maximising this as adversary wants to disciminate better
	adversary_objective = tf.reduce_mean(ll_data + ll_one_minus_fake)

	# Also, maximising this instead of minimizing mean(ll_one_minus_fake)
	generator_objective = tf.reduce_mean(ll_fake)

	return -adversary_objective, -generator_objective

def make(train_inference, test_inference):
	train_adversary_loss_op, train_generator_loss_op = loss(train_inference['ll_data'], train_inference['ll_fake'], train_inference['ll_one_minus_fake'])
	test_adversary_loss_op, test_generator_loss_op = loss(test_inference['ll_data'], test_inference['ll_fake'], test_inference['ll_one_minus_fake'])

	return {'train_adversary_loss': train_adversary_loss_op,
					'train_generator_loss': train_generator_loss_op,
					'test_adversary_loss': test_adversary_loss_op,
					'test_generator_loss': test_generator_loss_op}