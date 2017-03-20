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
def loss(lg_p_x_given_z, lg_p_z, lg_q_z_given_x):
	reconstruction_term = lg_p_x_given_z + lg_p_z
	regularizer_term = lg_q_z_given_x

	return -tf.reduce_mean(reconstruction_term - regularizer_term)

def make(train_inference, test_inference):
	train_loss_op = loss(train_inference['ll_decoder'], train_inference['ll_prior'], train_inference['ll_encoder'])
	test_loss_op = loss(test_inference['ll_decoder'], test_inference['ll_prior'], test_inference['ll_encoder'])

	return {'train_loss': train_loss_op, 'test_loss': test_loss_op}