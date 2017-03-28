﻿# MIT License
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
try:
	import cPickle as pickle
except:
	import pickle
import utils.file

# Save results so we can plot them later and resume from snapshots
def save_results(filepath, epoch, results, prior_noise):
	results_filepath = filepath + '-' + str(epoch) + '.results'
	with open(results_filepath, 'wb') as f:
		pickle.dump(results, f, protocol = pickle.HIGHEST_PROTOCOL)
		pickle.dump(prior_noise, f, protocol = pickle.HIGHEST_PROTOCOL)

def load_results(filepath):
	results_filepath = filepath + '.results'
	print(results_filepath)
	if not utils.file.exists(results_filepath):
		raise IOError('Results file at epoch {} does not exist'.format(epoch))

	with open(results_filepath, 'rb') as f:
		results = pickle.load(f)
		prior_noise = pickle.load(f)

	return results, prior_noise