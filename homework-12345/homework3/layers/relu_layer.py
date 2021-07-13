""" ReLU Layer """

import numpy as np
def Relu(x):
	y=np.copy(x)
	y[x<0]=0
	return y
def Relu_deriv(x):
	y=np.copy(x)
	y[x<0]=0
	y[x>0]=1
	return y	
class ReLULayer():
	def __init__(self):
		"""
		Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
		"""
		self.trainable = False # no parameters

	def forward(self, Input, **kwargs):

		############################################################################
	    # TODO: Put your code here
		# Apply ReLU activation function to Input, and return results.
		self.Input=Input
		return Relu(Input)
	    ############################################################################


	def backward(self, delta):
		

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta

		return np.multiply(delta,Relu_deriv(self.Input))
	    ############################################################################
