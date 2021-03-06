# -*- encoding: utf-8 -*-

import numpy as np

# if you implement ConvLayer by convolve function, you will use the following code.
from scipy.signal import fftconvolve as convolve
from scipy import signal
from scipy.signal.ltisys import freqresp 
def split_by_strides(X, kh, kw, s):
    N, H, W, C = X.shape
    oh = (H - kh) // s + 1
    ow = (W - kw) // s + 1
    shape = (N, oh, ow, kh, kw, C)
    strides = (X.strides[0], X.strides[1]*s, X.strides[2]*s, *X.strides[1:])
    A = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return A
class ConvLayer():
	"""
	2D convolutional layer.
	This layer creates a convolution kernel that is convolved with the layer
	input to produce a tensor of outputs.
	Arguments:
		inputs: Integer, the channels number of input.
		filters: Integer, the number of filters in the convolution.
		kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
		pad: Integer, the size of padding area.
		trainable: Boolean, whether this layer is trainable.
	"""
	def __init__(self, inputs,
	             filters,
	             kernel_size,
	             pad,
	             trainable=True):
		self.inputs = inputs
		self.filters = filters
		self.kernel_size = kernel_size
		self.pad = pad
		assert pad < kernel_size, "pad should be less than kernel_size"
		self.trainable = trainable

		self.XavierInit()


		self.grad_W = np.zeros_like(self.W)
		self.grad_b = np.zeros_like(self.b)

	def XavierInit(self):
		raw_std = (2 / (self.inputs + self.filters))**0.5
		init_std = raw_std * (2**0.5)

		self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))
		self.b = np.random.normal(0, init_std, (self.filters,))

	def forward(self, Input, **kwargs):
		'''
		forward method: perform convolution operation on the input.
		Agrs:
			Input: A batch of images, shape-(batch_size, channels, height, width)
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
		# try:
		self.Input = Input
		
		input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		self.input_after_pad=input_after_pad
		self.batch_size,self.channels,self.height,self.width=Input.shape

		input_after_trans=input_after_pad.transpose(0,2,3,1)
		self.input_after_trans=input_after_pad.transpose(0,2,3,1)
		# kh, kw, C, kn = self.filters.shape
		kh=self.kernel_size
		kw=self.kernel_size
		s=1
		# print(Input.shape)
		# print(inputs.shape)
		X_split = split_by_strides(input_after_trans, kh, kw, s)   # X_split.shape: (N, oh, ow, kh, kw, C)
		# print(X_split.shape)
		feature_map = np.tensordot(X_split,self.W, axes=[(3,4,5), (2,3,1)])
		# print(split_by_strides(inputs,kernel_size,kernel_size,1).shape)
		# print(feature_map.shape)

		return feature_map.transpose(0,3,1,2)#,self.W


	    ############################################################################


	def backward(self, delta):
		'''
		backward method: perform back-propagation operation on weights and biases.
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate self.grad_W, self.grad_b, and return the new delta.

		kh=self.kernel_size
		kw=self.kernel_size
		s=1
		X_split = split_by_strides(self.input_after_trans, kh, kw, s)   # X_split.shape: (N, oh, ow, kh, kw, C)
		self.grad_W = np.tensordot(X_split,delta, axes=[(0,1,2), (0,2,3)]).transpose(3,2,0,1)

		self.grad_W/=self.batch_size
		self.grad_b=delta.sum(axis=(0,2,3))/self.batch_size
		# self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))

		# shape-(batch_size, filters, output_height, output_width)
		pad=(self.kernel_size-1)//2
		delta_after_pad = np.pad(delta, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

		X_split = split_by_strides(delta_after_pad.transpose(0,2,3,1), kh, kw, s)   # X_split.shape: (N, oh, ow, kh, kw, C)
		local_delta = np.tensordot(np.flip(self.W,axis=(2,3)),X_split, axes=[(0,2,3), (5,3,4)])

		return local_delta.transpose(1,0,2,3)
	    ############################################################################
