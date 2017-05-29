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
def loss(loglike, D_fake, D_real, D_inter, Z_inter, X, name, scale, transform):	
	# Eq (3.9)
	# NOTE: Take negative since we are minimizing

	#print('lg_p_x_given_z.shape', lg_p_x_given_z.shape)
	#print('critic.shape', critic.shape)
	#print('prior_critic.shape', prior_critic.shape)
	#print('name', name)

	#print('*** DEBUG ***')
	#print('loglike.shape', loglike.shape)
	#print('D_fake.shape', D_fake.shape)
	#print('D_real.shape', D_real.shape)
	#print('D_inter.shape', D_inter.shape)
	#print('Z_inter.shape', Z_inter.shape)
	#print('X.shape', X.shape)

	# TODO: Flatten X? Or flatten grad2?
	lam = 10
	grad = tf.gradients(D_inter, [Z_inter])
	grad2 = tf.gradients(D_inter, [X])

	#print('X.name', X.name)

	#print('grad', grad)
	#print('grad2', grad2)

	#print('grad.shape', grad[0].shape)
	#print('grad2.shape', grad2[0].shape)

	grad_norm = tf.sqrt(tf.reduce_sum((tf.concat([grad[0], tf_models.flatten(grad2[0])], axis=1))**2, axis=1))
	grad_pen = lam * tf.reduce_mean(grad_norm - 1.)**2

	minus_EM = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
	D_loss = tf.reduce_mean(minus_EM + grad_pen)
	EM = tf.abs(minus_EM)
	rescaled_EM = EM * scale

	# TODO: Be able to change scaling factor in settings file!
	if transform == 'sqrt':
		regu_term = -tf.sqrt(tf.abs(minus_EM)) * scale
	elif transform == 'id':
		regu_term = -tf.abs(minus_EM) * scale
	elif transform == 'sq':
		regu_term = -tf.square(minus_EM) * scale
	elif transform == 'hypoth1':
		regu_term = -(tf.square(minus_EM) + tf.pow(tf.abs(minus_EM), 4) / 9.) * scale
	elif transform == 'hypoth2':
		regu_term = -(tf.square(minus_EM) + tf.pow(tf.abs(minus_EM), 4) / 18.) * scale
	elif transform == 'hypoth3':
		regu_term = -(tf.square(minus_EM) + tf.pow(tf.abs(minus_EM), 4) / 18.) * scale
	elif transform == 'hypoth4':
		regu_term = -tf.pow(tf.abs(minus_EM), 2.5) * scale
	elif transform == 'hypoth5':
		regu_term = -tf.pow(tf.abs(minus_EM), 3.) * scale
	elif transform == 'hypoth6':
		regu_term = -tf.pow(tf.abs(minus_EM), 4.) * scale
	elif transform == 'hypoth7':
		#regu_term = -(tf.abs(minus_EM) * tf.log((1 + tf.abs(minus_EM)) / (1 - tf.abs(minus_EM)))) * scale
		regu_term = -(tf.square(minus_EM) / 2. + tf.pow(tf.abs(minus_EM), 4) / 36. + tf.pow(tf.abs(minus_EM), 6) / 288.) * scale
	elif transform == 'hypoth8':
		regu_term = -(tf.square(minus_EM) / 2. + tf.pow(tf.abs(minus_EM), 4) / 36.) * scale
	elif transform == 'hypoth9':
		regu_term = -(1 + EM + tf.square(EM) / 2. + tf.pow(EM, 3) / 6. + tf.pow(EM, 4) / 24.) * scale
	elif transform == 'hypoth10':
		regu_term = -(tf.square(EM) / 2. + tf.pow(EM, 4.) / 24.) * scale
	elif transform == 'hypoth11':
		regu_term = -(tf.square(EM) / 2. + tf.pow(EM, 3) / 6. + tf.pow(EM, 4.) / 24.) * scale
	elif transform == 'hypoth12':
		regu_term = -(tf.square(EM) / 2. + tf.pow(EM, 3) / 6.) * scale
	elif transform == 'hypoth13':
		regu_term = -(tf.square(EM) / 2. + tf.pow(EM, 4.) / 24. + tf.pow(EM, 6.) / 120.) * scale
	elif transform == 'hypoth14':
		regu_term = -(tf.exp(EM * scale) - 1)
	elif transform == 'hypoth15':
		regu_term = -(rescaled_EM + tf.square(rescaled_EM) / 2. + tf.pow(rescaled_EM, 3.) / 6.)
	elif transform == 'hypoth16':
		regu_term = -(rescaled_EM + tf.square(rescaled_EM) / 2. + tf.pow(rescaled_EM, 3.) / 6. + tf.pow(EM, 4.) / 24.)
	elif transform == 'hypoth17':
		regu_term = -(rescaled_EM + tf.square(rescaled_EM)  + tf.pow(rescaled_EM, 3.))
	elif transform == 'hypoth18':
		regu_term = -(rescaled_EM + tf.square(rescaled_EM / 2.))
	elif transform == 'hypoth19':
		regu_term = -(tf.square(minus_EM) / 2. + tf.pow(tf.abs(minus_EM), 4) / 40.) * scale
	elif transform == 'hypoth20':
		regu_term = -(tf.square(minus_EM) / 2. + tf.pow(tf.abs(minus_EM), 4) / 50.) * scale
	elif transform == 'hypoth21':
		regu_term = -(tf.square(minus_EM) / 2. + tf.pow(tf.abs(minus_EM), 4) / 30.) * scale
	elif transform == 'hypoth22':
		regu_term = -(tf.square(minus_EM) / 2. + tf.pow(tf.abs(minus_EM), 3) / 36.) * scale

	#print('*** DEBUG ***')
	#print('loglike.shape', loglike.shape)
	#print('regu_term.shape', regu_term.shape)
	#raise Exception()

	elbo_loss = tf.reduce_mean(-loglike) - regu_term

	#discriminator_loss = -tf.reduce_mean(prior_critic - critic)
	#elbo_loss = -tf.reduce_mean(lg_p_x_given_z) + discriminator_loss	

	#return tf.identity(elbo_loss, name=name+'/elbo_like'), tf.identity(discriminator_loss, name=name+'/critic')
	return tf.identity(elbo_loss, name=name+'/elbo_like'), tf.identity(D_loss, name=name+'/critic'), tf.identity(-tf.reduce_mean(loglike), name=name+'/nll'), tf.identity(regu_term, name=name+'/regularizer')

def create(name='train', settings=None):
	#print('outputs', [op.name for op in tf.get_collection(tf_models.GraphKeys.OUTPUTS)])

	lg_p_x_given_z = tf_models.get_output(name + '/p_x_given_z/log_prob')
	critic = tf_models.get_output(name + '/critic/generator')
	prior_critic = tf_models.get_output(name + '/critic/prior')

	inter_critic = tf_models.get_output(name + '/critic/inter')
	z_inter = tf_models.get_output(name + '/z/interpolated')
	#x = tf_models.get_output(name + '/x')
	x = get_input(name)

	return loss(lg_p_x_given_z, critic, prior_critic, inter_critic, z_inter, x, name, settings['em_scale'], settings['em_transform'])

def get_input(name):
	ops = tf.get_collection(tf_models.GraphKeys.INPUTS)
	for op in ops:
		if 'inputs/'+name+'/samples' in op.name:
			return op
	raise ValueError('No loss operation with substring "{}" exists'.format(name))
