""" Dropout Layer """

import numpy as np
from numpy.lib.shape_base import tile

class DropoutLayer():
	def __init__(self,p=0.5):
		self.trainable = False
		self.p=p

	def forward(self, Input, is_training=True):

		############################################################################
	    # TODO: Put your code here
		child_shape=Input[0].shape
		self.dropout=np.random.choice(2,size=child_shape,p=(self.p,1-self.p))
		tile_shape=np.ones(len(Input.shape),dtype=np.int8)
		tile_shape[0]=Input.shape[0]
		self.final=np.tile(self.dropout,tile_shape)
		return np.multiply(self.final,Input)

	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		
		return np.multiply(self.final,delta)

	    ############################################################################
