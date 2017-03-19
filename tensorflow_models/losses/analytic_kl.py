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

# Loss for a basic VAE where ELBO has been rewritten and KL divergence has been analytically calculated
# TODO: Better understanding of loss function
# TODO: Can we rewrite this in terms of p_xz, q_z_given_x etc.?
def loss(x, x_reconstr_mean, z_mean, z_log_sigma_sq):
		# The loss is composed of two terms:
		# 1.) The reconstruction loss (the negative log probability
		#	 of the input under the reconstructed Bernoulli distribution 
		#	 induced by the decoder in the data space).
		#	 This can be interpreted as the number of "nats" required
		#	 for reconstructing the input when the activation in latent
		#	 is given.
		# Adding 1e-10 to avoid evaluation of log(0.0)
		reconstr_loss = \
			-tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean)
						   + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean),
						   1)

		# 2.) The latent loss, which is defined as the Kullback Leibler divergence 
		##	between the distribution in latent space induced by the encoder on 
		#	 the data and some prior. This acts as a kind of regularizer.
		#	 This can be interpreted as the number of "nats" required
		#	 for transmitting the the latent space distribution given
		#	 the prior.
		latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
										   - tf.square(z_mean)
										   - tf.exp(z_log_sigma_sq), 1)

		return tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch