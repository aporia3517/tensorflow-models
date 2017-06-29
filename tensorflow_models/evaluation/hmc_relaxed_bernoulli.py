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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import range

def kinetic_energy(velocity):
    return 0.5 * tf.reduce_sum(tf.multiply(velocity, velocity), axis=1)

def hamiltonian(position_z, position_eps, velocity_z, velocity_eps, log_posterior):
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
	energy_function = tf.squeeze(-log_posterior(position_z, position_eps))
	return energy_function + kinetic_energy(velocity_z) + kinetic_energy(velocity_eps)

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

def simulate_dynamics(initial_pos_z, initial_pos_eps, initial_vel_z, initial_vel_eps, stepsize, n_steps, log_posterior):
	def leapfrog(pos_z, pos_eps, vel_z, vel_eps, step, i):
		# TODO: Check whether reduce_sum is correct
		dE_dpos = tf.gradients(tf.squeeze(-log_posterior(pos_z, pos_eps)), [pos_z, pos_eps])	#[0]
		new_vel_z = vel_z - step * dE_dpos[0]
		new_pos_z = pos_z + step * new_vel_z

		new_vel_eps = vel_eps - step * dE_dpos[1]
		new_pos_eps = pos_eps + step * new_vel_eps

		return [new_pos_z, new_pos_eps, new_vel_z, new_vel_eps, step, tf.add(i, 1)]

	def condition(pos_z, pos_eps, vel_z, vel_eps, step, i):
		return tf.less(i, n_steps)

	dE_dpos = tf.gradients(tf.squeeze(-log_posterior(initial_pos_z, initial_pos_eps)), [initial_pos_z, initial_pos_eps])

	stepsize = tf.reshape(stepsize, [-1, 1])

	#print('*** critical point ***')
	#print('dE_dpos.shape', dE_dpos.shape)
	#print('stepsize.shape', stepsize.shape)
	#print('initial_vel.shape', initial_vel.shape)

	vel_half_step_z = initial_vel_z - 0.5 * tf.reshape(stepsize, [-1, 1]) * dE_dpos[0]
	pos_full_step_z = initial_pos_z + tf.reshape(stepsize, [-1, 1]) * vel_half_step_z

	vel_half_step_eps = initial_vel_eps - 0.5 * tf.reshape(stepsize, [-1, 1]) * dE_dpos[1]
	pos_full_step_eps = initial_pos_eps + tf.reshape(stepsize, [-1, 1]) * vel_half_step_eps

	#print('vel_half_step.shape', vel_half_step.shape)
	#print('pos_full_step.shape', pos_full_step.shape)
	#print('*** critical point ***')

	i = tf.constant(0)
	final_pos_z, final_pos_eps, new_vel_z, new_vel_eps, _, _ = tf.while_loop(condition, leapfrog, [pos_full_step_z, pos_full_step_eps, vel_half_step_z, vel_half_step_eps, stepsize, i], parallel_iterations=1)

	dE_dpos = tf.gradients(tf.squeeze(-log_posterior(final_pos_z, final_pos_eps)), [final_pos_z, final_pos_eps])

	final_vel_z = new_vel_z - 0.5 * stepsize * dE_dpos[0]
	final_vel_eps = new_vel_eps - 0.5 * stepsize * dE_dpos[1]

	return final_pos_z, final_pos_eps, final_vel_z, final_vel_eps

def hmc_step(initial_pos_z, initial_pos_eps, log_posterior, step_size, num_steps):
	initial_vel_z = tf.random_normal(tf.shape(initial_pos_z))
	initial_vel_eps = tf.random_normal(tf.shape(initial_pos_eps))

	final_pos_z, final_pos_eps, final_vel_z, final_vel_eps = simulate_dynamics(initial_pos_z, initial_pos_eps, initial_vel_z, initial_vel_eps, step_size, num_steps, log_posterior)

	#print('initial_pos.shape', initial_pos.shape)
	#print('initial_vel.shape', initial_vel.shape)
	#print('final_pos.shape', final_pos.shape)
	#print('final_vel.shape', final_vel.shape)
	#print('step_size.shape', step_size.shape)

	energy_prev = hamiltonian(initial_pos_z, initial_pos_eps, initial_vel_z, initial_vel_eps, log_posterior),
	energy_next = hamiltonian(final_pos_z, final_pos_eps, final_vel_z, final_vel_eps, log_posterior)
	accept = metropolis_hastings_accept(energy_prev, energy_next)

	#print('accept.shape', accept.shape)

	new_pos_z = tf.where(accept, final_pos_z, initial_pos_z)
	new_pos_eps = tf.where(accept, final_pos_eps, initial_pos_eps)

	return new_pos_z, new_pos_eps, accept

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
