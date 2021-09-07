from numpy.core.fromnumeric import size
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
import numpy as np

import sys
 
sys.path.append('../')
from layers import FCLayer, ReLULayer, ConvLayer, MaxPoolingLayer, ReshapeLayer
from layers.conv_layer_im2col import ConvLayer_im2col

from datetime import datetime
input_dim=2
channels=2
kernel_size=3
height=3
batch=3


inputs=np.random.randint(9,size=(batch,height,height,input_dim)).astype(np.float32)
# conv=tf.keras.layers.Conv2D(self.channels, (self.kernel_size, self.kernel_size), strides=(1, 1), input_shape=(self.height, self.width, 1), padding='same', activation='relu', kernel_initializer='uniform')
# offital_conv=tf.keras.layers.Conv2D(channels, (kernel_size, kernel_size), strides=(1, 1), input_shape=(height, height, input_dim), padding='same', activation='relu', kernel_initializer='uniform')
own_conv=ConvLayer(input_dim, channels, kernel_size, 1)
times=100

# shape-(batch_size, filters, output_height, output_width)

delta=np.random.randint(9,size=(batch,channels,height,height)).astype(np.float32)
print('own')
start=datetime.now()
for i in range(times):
    res2,kernel=own_conv.forward(inputs.transpose(0,3,1,2))
print('time: ',datetime.now()-start)
new_d=own_conv.backward(delta)
# print(res2.shape)
# print(res2)
print(new_d)




im_conv=ConvLayer_im2col(input_dim, channels, kernel_size, 1,kernel,kernel)


# shape-(batch_size, filters, output_height, output_width)

delta=np.random.randint(9,size=(batch,channels,height,height)).astype(np.float32)
print('im2col')
start=datetime.now()
for i in range(times):
    res3=im_conv.forward(inputs.transpose(0,3,1,2))
    im_conv.backward(delta)
print('time: ',datetime.now()-start)
# import pdb
# pdb.set_trace()
print('diff: ',np.sum(np.abs(res3-res2)))

print('own')
start=datetime.now()
for i in range(times):
    res2,kernel=own_conv.forward(inputs.transpose(0,3,1,2))
print('time: ',datetime.now()-start)

# h_s,w_s=res3.strides
# strides=    h_s,w_s*9,  h_s,w_s
# shape=(3,2,9)
# np.lib.stride_tricks.as_strided(res3, shape=shape, strides=strides)

# print('offitial')
# # kernel = np.array([[1, 0, -1], [1, 0, -2], [1, 0, -1]]).reshape((3, 3, 1, 1))
# print(kernel.shape)

# r = tf.nn.conv2d(input=inputs, filters=kernel.transpose(2,3,1,0), strides=[1, 1, 1, 1], padding="SAME")
# print(r.shape)
# # print((offital_conv(inputs)))

# sess=tf.Session()
# with sess.as_default():
#     print(r.eval())
#     res1=r.eval()
# print(res1.shape)
# print(res2)


# subs=(res2.transpose(0,2,3,1)-res1)
# print(subs)
# print(np.sum(subs))


# print()
# print()
# print()
# print(res2.transpose(0,2,3,1))
