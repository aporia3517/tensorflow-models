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