""" Fully Connected Layer """

import numpy as np
from layers.sigmoid_layer import sigmoid,sigmoid_deriv
from layers.relu_layer import Relu,Relu_deriv


class FCLayer():
	def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
		"""
		Apply a linear transformation to the incoming data: y = Wx + b
		Args:
			num_input: size of each input sample
			num_output: size of each output sample
			actFunction: the name of the activation function such as 'relu', 'sigmoid'
			trainable: whether if this layer is trainable
		"""
		self.num_input = num_input
		self.num_output = num_output
		self.trainable = trainable
		self.actFunction = actFunction
		assert actFunction in ['relu', 'sigmoid']

		self.XavierInit()

		self.grad_W = np.zeros((num_input, num_output))
		self.grad_b = np.zeros((1, num_output))


	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply linear transformation(Wx+b) to Input, and return results.
		self.Input=Input
		self.batch_size=Input.shape[0]
		if 'relu' == self.actFunction:
			return np.dot(Relu(Input),self.W)+self.b
		elif 'sigmoid' == self.actFunction:
			return np.dot(sigmoid(Input),self.W,)+self.b
	    ############################################################################


	def backward(self, delta):
		# The delta of this layer has been calculated in the later layer.
		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta
		if 'relu' == self.actFunction:
			hx=Relu(self.Input)
			hx_deriv=Relu_deriv(self.Input)
		elif 'sigmoid' == self.actFunction:
			hx=sigmoid(self.Input)
			hx_deriv=sigmoid_deriv(self.Input)
		for i in range(self.batch_size):
			self.grad_W+=np.dot(hx[i].reshape(self.num_input,1),delta[i].reshape(1,self.num_output))
		self.grad_W/=self.batch_size
		self.grad_b=np.sum(delta,axis=0).reshape(self.grad_b.shape)/self.batch_size
		local_delta=np.zeros((self.batch_size,self.num_input))
		# for i in range(self.num_input):
		for i in range(self.batch_size):

			local_delta[i]=np.multiply(np.dot(delta[i].reshape(1,self.num_output),self.W.transpose()),hx_deriv[i])
			# local_delta[i]=(np.dot(delta.transpose(),self.W.transpose()[:,i]))*np.sum(hx_deriv[:,i])

		return local_delta
	    ############################################################################


	def XavierInit(self):
		# Initialize the weigths according to the type of activation function.
		raw_std = (2 / (self.num_input + self.num_output))**0.5
		if 'relu' == self.actFunction:
			init_std = raw_std * (2**0.5)
		elif 'sigmoid' == self.actFunction:
			init_std = raw_std
		else:
			init_std = raw_std # * 4

		self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
		self.b = np.random.normal(0, init_std, (1, self.num_output))
