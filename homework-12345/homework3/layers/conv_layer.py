# -*- encoding: utf-8 -*-

import numpy as np

# if you implement ConvLayer by convolve function, you will use the following code.
from scipy.signal import fftconvolve as convolve
from scipy import signal 

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
		try:
			self.Input = Input
			
			input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
			self.batch_size,self.channels,self.height,self.width=Input.shape

			input_after_trans=input_after_pad.transpose(1,2,3,0)#  channels, height, width, batch_size
			self.input_after_pad=input_after_pad
			self.input_after_trans=input_after_trans
			output=np.zeros((self.filters,self.height,self.width,self.batch_size))
			# compute for each filter

			# import pdb 
			# pdb.set_trace()
			for i in range(self.filters):
				output[i]=signal.convolve(input_after_trans,np.flip(self.W[i][:,:,:,None],(0,1,2)),mode='valid')

			return output.transpose(3,0,1,2)
		except:
			import pdb 
			pdb.set_trace()




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

		for k in range(self.batch_size):
			for i in range(self.filters):
				self.grad_W[i]+=(signal.convolve(self.input_after_pad[k].transpose(1,2,0),np.flip(delta[k,i][:,:,None],(0,1,2)),mode='valid')).transpose(2,0,1)#.sum(axis=0)/self.batch_size
		self.grad_W[i]/=self.batch_size
		print(" success compute grad_w")
		self.grad_b=delta.sum(axis=(0,2,3))/self.batch_size
		print(" success compute grad_b")

		local_delta=np.zeros(self.Input.shape)#self.batch_size, self.filters, self.height, self.width)
		local_delta_re=local_delta.transpose(1,2,3,0)# self.filters, self.height, self.width, self.batch_size,self.batch_size
		pad=(self.W.shape[0]-1)//2
		for k in range(self.inputs):
			local_delta_re[k]=signal.convolve(np.flip(delta.transpose(1,2,3,0),axis=(0)),self.W.transpose(1,0,2,3)[k][:,:,:,None],mode='same')[pad]
			#signal.convolve(self.W.transpose(1,0,2,3)[k][:,:,:,None],np.flip(delta.transpose(1,2,3,0),axis=(0)),mode='same')[pad]#:-pad]
			
		local_delta=local_delta_re.transpose(3,0,1,2)
		print(" success compute delta")

		return local_delta
	    ############################################################################
