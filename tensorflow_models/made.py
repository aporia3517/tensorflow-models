# MIT License
#
# Copyright (c) 2017, Probabilistic Programming Group at University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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

import sys
import inspect

import os
import sys
import time
import six
import math
from six.moves import cPickle as pickle
from functools import reduce

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

import tensorflow_utils as tf_utils
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as dist

# Takes a vector as input and outputs the parameters to a mixture of Gaussians
# Wrap in a variable_scope context manager to create different Gaussian mixture models

# input_dimension should be number of latent variables
# parents_dimension should be number of observed variables => 784
# hidden_dimensions should be, e.g., [200, 200]
def make_made_logistic(inputs, parents, masks, input_dimension, parents_dimension, hidden_dimensions, K=3, activation_fn=tf.nn.relu):
    inputs = tf.concat([parents, inputs], 1)
    layer = inputs

    # Create the neural network using the weight masks
    # First create the hidden layers
    dimensions = [input_dimension + parents_dimension] + \
        hidden_dimensions  # + [input_dimension]
    for idx in range(len(hidden_dimensions)):
        with tf.variable_scope('layer_{}'.format(idx)):
            weights = tf.get_variable("weights", [dimensions[idx], dimensions[idx + 1]], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
            biases = tf.get_variable("biases", [dimensions[idx + 1]], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
            masked_weights = tf.multiply(weights, masks[idx])

            layer = activation_fn(tf.add(tf.matmul(layer, masked_weights), biases))

    # Create the output layer
    with tf.variable_scope('layer_{}'.format(len(hidden_dimensions))):
        # TODO: Change so get means/scales/coefs
        weights_logits = tf.get_variable("weights_logits", [dimensions[-1], input_dimension, K], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        skip_weights_logits = tf.get_variable("skip_weights_logits", [dimensions[0], input_dimension, K], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        biases_logits = tf.get_variable("biases_logits", [input_dimension, K], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

        weights_coefs = tf.get_variable("weights_coefs", [dimensions[-1], input_dimension, K], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        skip_weights_coefs = tf.get_variable("skip_weights_coefs", [dimensions[0], input_dimension, K], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        biases_coefs = tf.get_variable("biases_coefs", [input_dimension, K], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

    masked_weights_logits = tf.multiply(weights_logits, tf.expand_dims(masks[-2], -1))
    masked_skip_weights_logits = tf.multiply(skip_weights_logits, tf.expand_dims(masks[-1], -1))

    masked_weights_coefs = tf.multiply(weights_coefs, tf.expand_dims(masks[-2], -1))
    masked_skip_weights_coefs = tf.multiply(skip_weights_coefs, tf.expand_dims(masks[-1], -1))

    logits = tf.identity(tf.add(tf.add(tf.matmul(layer, tf.reshape(masked_weights_logits, (-1, K * input_dimension))), tf.reshape(biases_logits, (-1,))), tf.matmul(inputs, tf.reshape(masked_skip_weights_logits, (-1, K * input_dimension)))))
    logits = tf.reshape(logits, (-1, input_dimension, K))

    coefs = tf.nn.softmax(tf.add(tf.add(tf.matmul(layer, tf.reshape(masked_weights_coefs, (-1, K * input_dimension))), tf.reshape(biases_coefs, (-1,))), tf.matmul(inputs, tf.reshape(masked_skip_weights_coefs, (-1, K * input_dimension)))))
    coefs = tf.reshape(coefs, (-1, input_dimension, K))

    return logits, coefs

def make_made_logistic_single(inputs, parents, masks, input_dimension, parents_dimension, hidden_dimensions, activation_fn=tf.nn.relu):
    inputs = tf.concat([parents, inputs], 1)
    layer = inputs

    # Create the neural network using the weight masks
    # First create the hidden layers
    dimensions = [input_dimension + parents_dimension] + hidden_dimensions  # + [input_dimension]
    for idx in range(len(hidden_dimensions)):
        with tf.variable_scope('layer_{}'.format(idx)):
            weights = tf.get_variable("weights", [dimensions[idx], dimensions[idx + 1]], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
            biases = tf.get_variable("biases", [dimensions[idx + 1]], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
            masked_weights = tf.multiply(weights, masks[idx])

            layer = activation_fn(tf.add(tf.matmul(layer, masked_weights), biases))

    # Create the output layer
    with tf.variable_scope('layer_{}'.format(len(hidden_dimensions))):
        # TODO: Change so get means/scales/coefs
        weights_logits = tf.get_variable("weights_logits", [dimensions[-1], input_dimension, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        skip_weights_logits = tf.get_variable("skip_weights_logits", [dimensions[0], input_dimension, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        biases_logits = tf.get_variable("biases_logits", [input_dimension, 1], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

    masked_weights_logits = tf.multiply(weights_logits, tf.expand_dims(masks[-2], -1))
    masked_skip_weights_logits = tf.multiply(skip_weights_logits, tf.expand_dims(masks[-1], -1))

    logits = tf.identity(tf.add(tf.add(tf.matmul(layer, tf.reshape(masked_weights_logits, (-1, input_dimension))), tf.reshape(biases_logits, (-1,))), tf.matmul(inputs, tf.reshape(masked_skip_weights_logits, (-1, input_dimension)))))
    #logits = tf.reshape(logits, (-1, input_dimension))

    return logits

# Turns out this is the same as make_made for mixture model!
def make_made_categorical(inputs, parents, masks, input_dimension, parents_dimension, hidden_dimensions, count_categories=2, activation_fn=tf.nn.relu):
    if parents is not None:
        inputs = tf.concat([parents, inputs], 1)
    layer = inputs

    # Create the neural network using the weight masks
    # First create the hidden layers
    dimensions = [input_dimension*count_categories + parents_dimension] + hidden_dimensions  # + [input_dimension]
    for idx in range(len(hidden_dimensions)):
        with tf.variable_scope('layer_{}'.format(idx)):
            weights = tf.get_variable("weights", [dimensions[idx], dimensions[idx + 1]], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
            biases = tf.get_variable("biases", [dimensions[idx + 1]], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
            masked_weights = tf.multiply(weights, masks[idx])

            layer = activation_fn(tf.add(tf.matmul(layer, masked_weights), biases))

    # Create the output layer
    with tf.variable_scope('layer_{}'.format(len(hidden_dimensions))):
        # TODO: Change so get means/scales/coefs
        weights_logits = tf.get_variable("weights_logits", [dimensions[-1], input_dimension, count_categories], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        skip_weights_logits = tf.get_variable("skip_weights_logits", [dimensions[0], input_dimension, count_categories], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
        biases_logits = tf.get_variable("biases_logits", [input_dimension, count_categories], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

    masked_weights_logits = tf.multiply(weights_logits, tf.expand_dims(masks[-2], -1))
    masked_skip_weights_logits = tf.multiply(skip_weights_logits, tf.expand_dims(masks[-1], -1))

    #logits = tf.identity(tf.add(tf.add(tf.matmul(layer, tf.reshape(masked_weights_logits, (-1, input_dimension))), tf.reshape(biases_logits, (-1,))), tf.matmul(inputs, tf.reshape(masked_skip_weights_logits, (-1, input_dimension)))))
    logits = tf.identity(tf.add(tf.add(tf.matmul(layer, tf.reshape(masked_weights_logits, (-1, count_categories * input_dimension))), tf.reshape(biases_logits, (-1,))), tf.matmul(inputs, tf.reshape(masked_skip_weights_logits, (-1, count_categories * input_dimension)))))
    logits = tf.reshape(logits, (-1, input_dimension, count_categories))

    return logits

def made_masks(input_dimension, parents_dimension, hidden_dimensions, count_categories=1):
    masks = []
    maximum_inputs = []

    # Create maximum input arrays
    maximum_inputs.append(np.concatenate(
        (np.zeros(parents_dimension), np.repeat(np.arange(1, input_dimension + 1), count_categories))))
    for d in hidden_dimensions:
        maximum_inputs.append(np.random.random_integers(low=np.amin(
            maximum_inputs[-1]), high=input_dimension - 1, size=d))
    maximum_inputs.append(np.arange(1, input_dimension + 1))

    # MADE for categorical distributions
    #print(maximum_inputs[0])
    #raise Exception()

    # Create masks between input layer and hidden layers
    for idx in range(len(hidden_dimensions)):
        masks.append((maximum_inputs[idx + 1].reshape(
            (-1, 1)) >= maximum_inputs[idx]).astype(dtype=np.float32))

    #print(maximum_inputs[0].shape, maximum_inputs[-1].reshape((-1, 1)).shape)
    #print((maximum_inputs[0] > maximum_inputs[-1].reshape((-1, 1))).shape)
    #raise Exception()

    # Create mask to output layer from last hidden layer
    masks.append(np.transpose(maximum_inputs[-1] > maximum_inputs[-2]
                 .reshape((-1, 1))).astype(dtype=np.float32))

    # Create mask from input to output layer
    masks.append(np.transpose(maximum_inputs[-1] > maximum_inputs[0]
                 .reshape((-1, 1))).astype(dtype=np.float32))

    return masks