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

import matplotlib
matplotlib.use('Agg')   # NOTE: This is the backend that does PNG files
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from six.moves import range

# TODO: More extensive set of functions to visualize learning, both during and afterwards

def sample_grid(outputfilename, samples, sample_size, ext = 'png', imgrange = (0.0, 1.0)):
	range_min, range_max = imgrange
	n, h, w, c = samples.shape
	assert(n >= sample_size)
	n = sample_size

	# DEBUG
	#print('imgrange', imgrange)
	#print('max sample', np.max(samples))
	#print('min sample', np.min(samples))

	# Normalize samples so between 0 and 1
	samples = (samples - range_min) / np.float((range_max - range_min))
	samples_pr_side = int(np.sqrt(n))

	plt.figure(figsize = (7, 7), dpi=80)

	# Greyscale images
	if c == 1:
		canvas = np.zeros((h * samples_pr_side, samples_pr_side * w))
		idx = 0
		for i in range(samples_pr_side):
			for j in range(samples_pr_side):
				canvas[i*h:(i+1)*h, j*w:(j+1)*w] = np.clip(samples[idx, :, :, 0], 1e-6, 1-1e-6)
				idx += 1
		plt.imshow(canvas, cmap = 'gray')

	# Colour images
	if c == 3:
		canvas = np.zeros((h * samples_pr_side, samples_pr_side * w, 3))
		idx = 0
		for i in range(samples_pr_side):
			for j in range(samples_pr_side):
				canvas[i*h:(i+1)*h, j*w:(j+1)*w, 0] = np.clip(samples[idx, :, :, 0], 1e-6, 1-1e-6)
				canvas[i*h:(i+1)*h, j*w:(j+1)*w, 1] = np.clip(samples[idx, :, :, 1], 1e-6, 1-1e-6)
				canvas[i*h:(i+1)*h, j*w:(j+1)*w, 2] = np.clip(samples[idx, :, :, 2], 1e-6, 1-1e-6)
				idx += 1
		plt.imshow(canvas)

	plt.savefig(outputfilename + '.' + ext, dpi=80)
	plt.close()

def plot_codes(filepath, codes, ext='png'):
	epoch_count, latent_dimension = codes.shape

	# Normalize codes
	max_val = np.max(codes)
	min_val = np.min(codes)
	range_val = max_val - min_val
	codes = (codes - min_val) / range_val

	fig = plt.figure(figsize = (latent_dimension / 2., epoch_count), dpi=80)

	for idx in range(epoch_count):
		ax = fig.add_subplot(epoch_count, 1, idx + 1)
		ax.imshow(np.reshape(codes[idx, :], (1, latent_dimension)), cmap = 'gray')
		ax.axis('off')

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

def plot_reconstructions(filepath, truth, reconstructions, ext='png'):
	step_count, _, _, _ = reconstructions.shape

	# Normalize codes
	#max_val = np.max(reconstructions)
	#min_val = np.min(reconstructions)
	#range_val = max_val - min_val
	#reconstructions = (reconstructions - min_val) / range_val

	fig = plt.figure(figsize = (2, step_count), dpi=80)

	for idx in range(step_count):
		ax = fig.add_subplot(step_count, 2, 2*idx + 1)
		ax.imshow(np.reshape(truth, (32, 32)), cmap = 'gray')
		ax.axis('off')

		ax2 = fig.add_subplot(step_count, 2, 2*idx + 2)
		ax2.imshow(np.reshape(reconstructions[idx], (32, 32)), cmap = 'gray')
		ax2.axis('off')

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

def gan_experiment(filepath, title, ys, ext='png'):
	x = np.arange(1, len(ys[0]) + 1, dtype=np.int32)

	fig = plt.figure(figsize = (9.5, 5), dpi=80)
	fig.suptitle(title)

	ax1 = fig.add_subplot(1, 2, 1, title='generator')
	ax1.plot(x, ys[1])

	ax2 = fig.add_subplot(1, 2, 2, title='discriminator')
	ax2.plot(x, ys[0])

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

def wgan_experiment(filepath, title, ys, ext='png'):
	x = np.arange(1, len(ys[0]) + 1, dtype=np.int32)

	fig = plt.figure(figsize = (9.5, 5), dpi=80)
	fig.suptitle(title)

	ax1 = fig.add_subplot(1, 2, 1, title='generator')
	ax1.plot(x, ys[1])

	ax2 = fig.add_subplot(1, 2, 2, title='critic')
	ax2.plot(x, ys[0])

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

def avb_experiment(filepath, title, ys, ext='png'):
	x = np.arange(1, len(ys[0]) + 1, dtype=np.int32)

	fig = plt.figure(figsize = (9.5, 5), dpi=80)
	fig.suptitle(title)

	ax1 = fig.add_subplot(1, 2, 1, title='elbo-like')
	ax1.plot(x, ys[1])

	ax2 = fig.add_subplot(1, 2, 2, title='discriminator')
	ax2.plot(x, ys[0])

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

def emvb_experiment(filepath, title, ys, ext='png'):
	x = np.arange(1, len(ys[0]) + 1, dtype=np.int32)

	fig = plt.figure(figsize = (9.5, 5), dpi=80)
	fig.suptitle(title)

	ax1 = fig.add_subplot(1, 2, 1, title='elbo-like')
	ax1.plot(x, ys[1])

	ax2 = fig.add_subplot(1, 2, 2, title='critic')
	ax2.plot(x, ys[0])

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

# Plot AIS for an EMVB experiment
def emvb_ais(filepath, title, epochs_ys, ys, epochs_ais, ais, ext='png'):
	#x = np.arange(1, len(ys[0]) + 1, dtype=np.int32)

	fig = plt.figure(figsize = (9.5, 5), dpi=80)
	fig.suptitle(title)

	ax1 = fig.add_subplot(1, 2, 1, title='bounds')
	ax1.plot(epochs_ys, ys[1], label='ELBO-like', color='red')
	ax1.plot(epochs_ais, ais, label='AIS', color='blue')
	ax1.legend()

	ax2 = fig.add_subplot(1, 2, 2, title='critic')
	ax2.plot(epochs_ys, ys[0])

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

# Plot AIS for an AVB experiment
def avb_ais(filepath, title, epochs_ys, ys, epochs_ais, ais, ext='png'):
	#x = np.arange(1, len(ys[0]) + 1, dtype=np.int32)

	fig = plt.figure(figsize = (9.5, 5), dpi=80)
	fig.suptitle(title)

	ax1 = fig.add_subplot(1, 2, 1, title='bounds')
	ax1.plot(epochs_ys, ys[1], label='ELBO', color='red')
	ax1.plot(epochs_ais, ais, label='AIS', color='blue')
	ax1.legend()

	ax2 = fig.add_subplot(1, 2, 2, title='discriminator')
	ax2.plot(epochs_ys, ys[0])

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

# Plot AIS for an SVB experiment
def svb_ais(filepath, title, epochs_ys, ys, epochs_ais, ais, ext='png'):
	fig = plt.figure(figsize = (5, 5), dpi=80)
	fig.suptitle(title)

	ax1 = fig.add_subplot(1, 1, 1, title='bounds')
	ax1.plot(epochs_ys, ys, label='ELBO', color='red')
	ax1.plot(epochs_ais, ais, label='AIS', color='blue')
	ax1.legend()

	fig.savefig(filepath + '.' + ext, dpi=80)
	plt.close(fig)

def experiment(outputfilename, title, y, x = None, ylim = (0., 1.0), ext='png'):
	if x is None:
		x = np.arange(0, len(y), dtype=np.int32)

	#print('length of x:', len(x))
	#print('shape of y:', y.shape)

	plt.figure(figsize = (10, 10))
	plt.plot(x, y)
	plt.xlabel('epochs')
	plt.ylabel('nll bound')
	plt.title(title)
	plt.ylim(ylim)

	plt.savefig(outputfilename + '.' + ext)
	plt.close()

def experiments(outputfilename, title, y, x = None, ylim = (0., 1.0), ext = 'png'):	
	if x is None:
		x = np.arange(0, len(y), dtype=np.int32)

	#print('length of x:', len(x))
	#print('shape of y:', y.shape)

	plt.figure(figsize = (10, 10))
	plt.plot(x, y)
	plt.xlabel('epochs')
	plt.ylabel('nll bound')
	plt.title(title)
	plt.ylim(ylim)

	plt.savefig(outputfilename + '.' + ext)
	plt.close()
