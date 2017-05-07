from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import range

import tensorflow_models as tf_models
from tensorflow_models.evaluation.hmc import hmc_step, hmc_updates

def ais(sample_prior_op, lg_prior, lg_px_given_z, x, num_steps=20, count_intermediate=100):

	step_size = tf.Variable(tf.constant(0.1, shape=(x.shape[0],)), trainable=False)
	avg_acceptance_rate = tf.Variable(tf.constant(0.9, shape=(x.shape[0],)), trainable=False)

	def update(old_z, old_w, step_size, avg_acceptance_rate, i):
		# Construct the log-prior and scaled log-likelihood terms
		lg_likelihood = lambda z: lg_px_given_z(x, z) 
		scaled_lg_likelihood = lambda z: lg_likelihood(z) * (i + 1) / tf.constant(count_intermediate, dtype=tf.float32)
		intermediate_distribution = lambda z: lg_prior(z) + scaled_lg_likelihood(z)

		# Generate the sample z_i from z_{i-1} using an HMC transition
		new_z, accept_op = hmc_step(initial_pos=old_z, log_posterior=intermediate_distribution, step_size=step_size, num_steps=num_steps)
		new_step_size, new_avg_acceptance_rate = hmc_updates(accept_op, step_size, avg_acceptance_rate)

		# DEBUG
		#print('old_z.shape', new_z.shape)
		#print('new_z.shape', new_z.shape)
		#print('accept_op.shape', accept_op.shape)
		#print('new_step_size.shape', new_step_size.shape)
		#print('new_avg_acceptance_rate.shape', new_avg_acceptance_rate.shape)
		
		# Update the weights by adding 1/N lg p(x | z_{i-1})
		new_w = old_w + tf.reshape(lg_likelihood(old_z), (-1, 1)) / tf.constant(count_intermediate, dtype=tf.float32)

		#print('new_w', new_w.shape)

		return [new_z, new_w, new_step_size, new_avg_acceptance_rate, tf.add(i, 1)]
		#return [new_z, new_w, step_size, avg_acceptance_rate, tf.add(i, 1)]

	def condition(z, w, s, a, i):
		return tf.less(i, count_intermediate)

	# Generate z_0 from the prior, p(z)
	# TODO: Correct function to sample prior from model
	z = sample_prior_op

	# Initialize the weights to zero
	w = tf.zeros((x.shape[0], 1), dtype=tf.float32)
	i = tf.constant(0., dtype=tf.float32)

	# DEBUG
	#print('step_size.shape', step_size.shape)
	#print('avg_acceptance_rate.shape', avg_acceptance_rate.shape)
	#print('z.shape', z.shape)
	#print('w.shape', w.shape)

	final_z, final_w, new_step_size, new_acceptance_rate, _ = tf.while_loop(condition, update, [z, w, step_size, avg_acceptance_rate, i], parallel_iterations=1)

	return final_z, final_w, new_step_size, new_acceptance_rate
	#return final_z, final_w