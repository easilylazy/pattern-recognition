""" Softmax Cross-Entropy Loss Layer """

import numpy as np
from layers.sigmoid_layer import sigmoid
# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11



class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		self.logit=logit
		self.gt=gt
		test_pred=np.argmax(logit,axis=1)
		test_true=np.argmax(gt,axis=1)
		err=(test_pred-test_true)
		self.acc = err[err==0].size/logit.shape[0]
		self.loss = -np.sum(np.multiply(gt,np.log(sigmoid(logit))))/logit.shape[0]

	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		return np.multiply(self.gt,(sigmoid(self.logit)-1))

	    ############################################################################
