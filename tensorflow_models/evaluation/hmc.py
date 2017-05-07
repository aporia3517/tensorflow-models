from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import range

def kinetic_energy(velocity):
    return 0.5 * tf.reduce_sum(tf.multiply(velocity, velocity), axis=1)

def hamiltonian(position, velocity, log_posterior):
	"""Computes the Hamiltonian of the current position, velocity pair
		H = U(x) + K(v)
		U is the potential energy and is = -log_posterior(x)
		Parameters
		----------
		position : tf.Variable
			Position or state vector x (sample from the target distribution)
		velocity : tf.Variable
			Auxiliary velocity variable
		energy_function
			Function from state to position to 'energy'
			= -log_posterior
		Returns
		-------
		hamitonian : float
		"""
	#print('position.shape', position.shape)
	#print('velocity.shape', position.shape)
	energy_function = tf.squeeze(-log_posterior(position))
	return energy_function + kinetic_energy(velocity)

def metropolis_hastings_accept(energy_prev, energy_next):
	#ediff = energy_prev - energy_next
	ediff = tf.squeeze(tf.subtract(energy_prev, energy_next))
	#ediff = tf.subtract(energy_prev, energy_next)

	#print('energy_prev', [e.shape for e in energy_prev])
	#print('energy_prev', energy_prev)
	#print('energy_next.shape', energy_next.shape)

	#print('energy_prev.shape', energy_prev.shape)
	#print('energy_next.shape', energy_next.shape)
	#print('ediff.shape', ediff.shape)

	#print('tf.exp(ediff).shape', tf.exp(ediff).shape)

	return (tf.exp(ediff) - tf.random_uniform(tf.shape(ediff))) >= 0.0

def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, log_posterior):
	def leapfrog(pos, vel, step, i):
		# TODO: Check whether reduce_sum is correct
		dE_dpos = tf.gradients(tf.squeeze(-log_posterior(pos)), pos)[0]
		new_vel = vel - step * dE_dpos
		new_pos = pos + step * new_vel
		return [new_pos, new_vel, step, tf.add(i, 1)]

	def condition(pos, vel, step, i):
		return tf.less(i, n_steps)

	dE_dpos = tf.gradients(tf.squeeze(-log_posterior(initial_pos)), initial_pos)[0]

	stepsize = tf.reshape(stepsize, [-1, 1])

	#print('*** critical point ***')
	#print('dE_dpos.shape', dE_dpos.shape)
	#print('stepsize.shape', stepsize.shape)
	#print('initial_vel.shape', initial_vel.shape)

	vel_half_step = initial_vel - 0.5 * tf.reshape(stepsize, [-1, 1]) * dE_dpos
	pos_full_step = initial_pos + tf.reshape(stepsize, [-1, 1]) * vel_half_step

	#print('vel_half_step.shape', vel_half_step.shape)
	#print('pos_full_step.shape', pos_full_step.shape)
	#print('*** critical point ***')

	i = tf.constant(0)
	final_pos, new_vel, _, _ = tf.while_loop(condition, leapfrog, [pos_full_step, vel_half_step, stepsize, i], parallel_iterations=1)

	dE_dpos = tf.gradients(tf.squeeze(-log_posterior(final_pos)), final_pos)[0]
	final_vel = new_vel - 0.5 * stepsize * dE_dpos
	return final_pos, final_vel

def hmc_step(initial_pos, log_posterior, step_size, num_steps):
	initial_vel = tf.random_normal(tf.shape(initial_pos))

	final_pos, final_vel = simulate_dynamics(initial_pos, initial_vel, step_size, num_steps, log_posterior)

	#print('initial_pos.shape', initial_pos.shape)
	#print('initial_vel.shape', initial_vel.shape)
	#print('final_pos.shape', final_pos.shape)
	#print('final_vel.shape', final_vel.shape)
	#print('step_size.shape', step_size.shape)

	energy_prev = hamiltonian(initial_pos, initial_vel, log_posterior),
	energy_next = hamiltonian(final_pos, final_vel, log_posterior)
	accept = metropolis_hastings_accept(energy_prev, energy_next)

	#print('accept.shape', accept.shape)

	new_pos = tf.where(accept, final_pos, initial_pos)

	return new_pos, accept

def hmc_updates(
	accept,
	stepsize,
	avg_acceptance_rate,
	target_acceptance_rate=0.9,
	stepsize_inc=1.02,
	stepsize_dec=0.98,
	stepsize_min=0.0001,
	stepsize_max=0.5,
	avg_acceptance_slowness=0.9):

	# DEBUG
	#print('*** Critical part ***')
	#print('stepsize.shape', stepsize.shape)
	#print('accept.shape', accept.shape)
	#print('avg_acceptance_rate.shape', avg_acceptance_rate.shape)

	new_stepsize_ = tf.where(avg_acceptance_rate > target_acceptance_rate, stepsize_inc*stepsize, stepsize_dec*stepsize)
	#print('new_stepsize_.shape', new_stepsize_.shape)

	new_stepsize = tf.maximum(tf.minimum(new_stepsize_, stepsize_max), stepsize_min)
	#print('new_stepsize.shape', new_stepsize.shape)

	#new_acceptance_rate = tf.add(avg_acceptance_slowness * avg_acceptance_rate, (1.0 - avg_acceptance_slowness) * tf.reduce_mean(tf.to_float(accept)))
	new_acceptance_rate = tf.add(avg_acceptance_slowness * avg_acceptance_rate, (1.0 - avg_acceptance_slowness) * tf.to_float(accept))
	#print('new_acceptance_rate.shape', new_acceptance_rate.shape)
	#print('*** Critical part ***')

	return new_stepsize, new_acceptance_rate