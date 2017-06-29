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

import six
import numpy as np
from six.moves import cPickle as pickle

import tensorflow_datasets as tf_data
import tensorflow_datasets.utils.file

# Save results so we can plot them later and resume from snapshots
def save_results(filepath, epoch, results):
	results_filepath = filepath + '-' + str(epoch) + '.results'
	with open(results_filepath, 'wb') as f:
		pickle.dump(results, f, protocol = pickle.HIGHEST_PROTOCOL)

# Save results so we can plot them later and resume from snapshots
def save_ais(filepath, settings, ais, ais_epochs):
	results_filepath = filepath + '-' + str(settings['ais_start']) + '-' + str(settings['ais_end']) + '.results'
	with open(results_filepath, 'wb') as f:
		pickle.dump((ais, ais_epochs), f, protocol = pickle.HIGHEST_PROTOCOL)

def load_ais(filepath):
	results_filepath = filepath + '.results'
	#print(results_filepath)
	if not tf_data.utils.file.exists(results_filepath):
		raise IOError('Results file at epoch {} does not exist'.format(epoch))

	with open(results_filepath, 'rb') as f:
		ais, ais_epochs = pickle.load(f)
		#print(aiss.keys())
		#ais, ais_epochs = aiss
		return ais, ais_epochs

def load_results(filepath):
	results_filepath = filepath + '.results'
	#print(results_filepath)
	#print(results_filepath)
	if not tf_data.utils.file.exists(results_filepath):
		raise IOError('Results file at epoch {} does not exist'.format(epoch))

	with open(results_filepath, 'rb') as f:
		results = pickle.load(f)
		return results
