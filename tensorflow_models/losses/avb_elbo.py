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

# Adversarial variational Bayes loss

# lg_p_x_given_z ~ batch_size x 784
# adversary ~ ?
# prior_adversary ~ ?
def loss(lg_p_x_given_z, discriminator, prior_discriminator):	
	# Eq (3.9)
	# NOTE: Take negative since we are minimizing
	elbo_loss = -tf.reduce_mean(-adversary + lg_p_x_given_z)

	# Eq (3.3)
	discriminator_loss = -tf.reduce_mean(tf_models.safe_log(tf.nn.sigmoid(adversary)) + tf_models.safe_log(1. - tf.nn.sigmoid(prior_adversary)))

	return tf.identity(elbo_loss, name=name+'/elbo_like'), tf.identity(discriminator_loss, name=name+'/discriminator')

def create(name='train'):
	lg_p_x_given_z = tf_models.outputs(name + '/p_x_given_z/log_prob')
	discriminator = tf_models.outputs(name + '/discriminator')
	prior_discriminator = tf_models.outputs(name + '/prior_discriminator')

	return loss(lg_p_x_given_z, discriminator, prior_discriminator, name=name)