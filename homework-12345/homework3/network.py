""" Network Class """

class Network():
	def __init__(self):
		self.layerList = []
		self.numLayer = 0
		self.is_training = True

	def add(self, layer):
		self.numLayer += 1
		self.layerList.append(layer)

	def forward(self, x):
		# forward layer by layer
		for i in range(self.numLayer):
			print('x shape')
			print(x.shape)
			x = self.layerList[i].forward(x, is_training=self.is_training)
			print('finist layer '+str(i))
		return x

	def backward(self, delta):
		# backward layer by layer
		for i in reversed(range(self.numLayer)): # reversed
			print('delta shape')
			print(delta.shape)
			delta = self.layerList[i].backward(delta)
			print('finist layer '+str(i))

