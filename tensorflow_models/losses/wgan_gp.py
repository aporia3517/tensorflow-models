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

def loss(critic_real, critic_fake, critic_interpolated, interpolated, name):
	lam = 10.
	grad = tf.gradients(critic_interpolated, [interpolated])[0]

	#print(grad)

	gradient_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))	
	gradient_penalty = lam * tf.reduce_mean(gradient_norm - 1.)**2

	critic_objective = tf.reduce_mean(critic_fake - critic_real) + gradient_penalty

	# Also, maximising this instead of minimizing mean(ll_one_minus_fake)
	generator_objective = -tf.reduce_mean(critic_fake)

	return tf.identity(critic_objective, name=name+'/critic'), tf.identity(generator_objective, name=name+'/generator')

def create(name='train'):
	critic_real = tf_models.get_output(name + '/p_x/critic_real')
	critic_fake = tf_models.get_output(name + '/p_x/critic_fake')
	critic_interpolated = tf_models.get_output(name + '/p_x/critic_interpolated')
	interpolated = tf_models.get_output(name + '/x/interpolated')

	return loss(critic_real, critic_fake, critic_interpolated, interpolated, name=name)