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

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply ReLU activation function to Input, and return results.
		y=np.copy(Input)
		y[Input<0]=0
		return y
	    ############################################################################


	def backward(self, delta):
		pass

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta


	    ############################################################################
