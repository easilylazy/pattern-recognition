""" Sigmoid Layer """

import numpy as np
EPS = 1e-11

def sigmoid(y):
	return 1/(1+1/(np.exp(y)+EPS))
def sigmoid_deriv(y):
	return sigmoid(y)*(1-sigmoid(y))
class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.
		self.Input=Input
		return sigmoid(Input)
	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta

		return np.multiply(delta,sigmoid_deriv(self.Input))
	    ############################################################################
