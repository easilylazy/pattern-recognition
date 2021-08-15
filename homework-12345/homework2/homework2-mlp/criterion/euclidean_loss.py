""" Euclidean Loss Layer """

import numpy as np
def sigmoid(y):
	return 1/(1+np.exp(-y))
def sigmoid_deriv(y):
	return sigmoid(y)*(1-sigmoid(y))
class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.

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
		self.res=sigmoid(self.logit)
		self.res_derive=self.res*(1-self.res)
		test_pred=np.argmax(logit,axis=1)
		test_true=np.argmax(gt,axis=1)
		err=(test_pred-test_true)
		self.acc = err[err==0].size/logit.shape[0]
		# print('accuracy: {0:f}'.format(self.acc))
		self.loss = np.sum(np.power(self.res-gt,2))/logit.shape[0]


	    ############################################################################

		return self.loss

	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)

		return (self.logit-self.gt)*self.res_derive
	    ############################################################################
