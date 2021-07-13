# -*- encoding: utf-8 -*-

import numpy as np

class MaxPoolingLayer():
	def __init__(self, kernel_size, pad):
		'''
		This class performs max pooling operation on the input.
		Args:
			kernel_size: The height/width of the pooling kernel.
			pad: The width of the pad zone.
		'''

		self.kernel_size = kernel_size
		self.pad = pad
		self.trainable = False

	def forward(self, Input, **kwargs):
		'''
		This method performs max pooling operation on the input.
		Args:
			Input: The input need to be pooled.
			 shape-(batch_size, channels, height, width)
		Return:
			The tensor after being pooled.
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
		try:
			self.Input = Input
			self.batch_size,self.channels,self.height,self.width=Input.shape
			input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)

			b_str,c_str,h_str,w_str=input_after_pad.strides
			# strides2=(c.strides[0]*k.shape[0],k.strides[0],c.strides[0],c.strides[1],)#c.strides[2],c.strides[1],c.strides[2])

			strides=(b_str,c_str,h_str*self.kernel_size,int(w_str/self.width)*self.kernel_size,h_str,w_str)
			shape = (self.batch_size, self.channels, int(self.height / self.kernel_size), int(self.width / self.kernel_size), self.kernel_size, self.kernel_size)
			return np.lib.stride_tricks.as_strided(input_after_pad,shape=shape,strides=strides).max(axis=(-2,-1))
		except:
			import pdb 
			pdb.set_trace()
	    ############################################################################

	def backward(self, delta):
		'''
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the new delta.
		pass

	    ############################################################################
