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

import tensorflow as tf

def training(loss, learning_rate=0.001, var_list=None, step=None, clip_grads=False, name='Adam', beta1=0.9, beta2=0.999):
	# Add a scalar summary for the snapshot loss.
	# TODO: Adding summaries!
	#tf.summary.scalar('loss', loss)

	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name=name, beta1=beta1, beta2=beta2)

	# Use the optimizer to apply the gradients that minimize the loss
	# TODO: Better way to code this??
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	# DEBUG
	print('update_ops', update_ops)

	with tf.control_dependencies(update_ops):
		if step is None:
			if var_list is None:
				train_op = optimizer.minimize(loss)
			else:
				train_op = optimizer.minimize(loss, var_list=var_list)
		else:
			if var_list is None:
				train_op = optimizer.minimize(loss, global_step=step)
			else:
				train_op = optimizer.minimize(loss, global_step=step, var_list=var_list)

	# TODO: Clip gradients!

	return train_op