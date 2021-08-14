from numpy.core.fromnumeric import size
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import numpy as np


from layers import FCLayer, ReLULayer, ConvLayer, MaxPoolingLayer, ReshapeLayer

input_dim=1
channels=1
kernel_size=2
height=4
batch=3


inputs=np.random.randint(9,size=(batch,height,height,input_dim)).astype(np.float32)
# conv=tf.keras.layers.Conv2D(self.channels, (self.kernel_size, self.kernel_size), strides=(1, 1), input_shape=(self.height, self.width, 1), padding='same', activation='relu', kernel_initializer='uniform')
# offital_conv=tf.keras.layers.Conv2D(channels, (kernel_size, kernel_size), strides=(1, 1), input_shape=(height, height, input_dim), padding='same', activation='relu', kernel_initializer='uniform')

own_pool=(MaxPoolingLayer(2, 0))


# shape-(batch_size, filters, output_height, output_width)
height_o=int(height / kernel_size)
			# self.width_o=int(self.width / self.kernel_size)
delta=np.random.randint(9,size=(batch,channels,height_o,height_o)).astype(np.float32)
print('own')
res=own_pool.forward(inputs.transpose(0,3,1,2))
# for input in inputs:
print(res)

print(inputs.shape)
print(res.shape)
new_delta=own_pool.backward(delta)
# # print(res2.shape)
print(new_delta)

