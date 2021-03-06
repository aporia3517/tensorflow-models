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
import tensorflow_models as tf_models

# The basic ELBO loss
def loss(lg_p_x_given_z, lg_p_z, lg_q_z_given_x, lg_p_x_given_z_discrete, lg_p_z_discrete, lg_q_z_given_x_discrete, name):
	reconstruction_term = lg_p_x_given_z + lg_p_z
	regularizer_term = lg_q_z_given_x
	elbo = tf.identity(-tf.reduce_mean(reconstruction_term - regularizer_term), name=name+'/elbo')

	reconstruction_term = lg_p_x_given_z_discrete + lg_p_z_discrete
	regularizer_term = lg_q_z_given_x_discrete
	elbo_discrete = tf.identity(-tf.reduce_mean(reconstruction_term - regularizer_term), name=name+'/elbo_discrete')

	return elbo, elbo_discrete

def create(name='train', settings=None):
	lg_p_x_given_z = tf_models.get_output(name + '/p_x_given_z/log_prob')
	lg_p_z = tf_models.get_output(name + '/p_z/log_prob')
	lg_q_z_given_x = tf_models.get_output(name + '/q_z_given_x/log_prob')

	lg_p_x_given_z_discrete = tf_models.get_output(name + '/p_x_given_z/log_prob_discrete')
	lg_p_z_discrete = tf_models.get_output(name + '/p_z/log_prob_discrete')
	lg_q_z_given_x_discrete = tf_models.get_output(name + '/q_z_given_x/log_prob_discrete')

	return loss(lg_p_x_given_z, lg_p_z, lg_q_z_given_x, lg_p_x_given_z_discrete, lg_p_z_discrete, lg_q_z_given_x_discrete, name=name)
