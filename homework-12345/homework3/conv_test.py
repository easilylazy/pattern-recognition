from numpy.core.fromnumeric import size
import tensorflow.compat.v1 as tf
import numpy as np


from layers import FCLayer, ReLULayer, ConvLayer, MaxPoolingLayer, ReshapeLayer

input_dim=1
channels=8
kernel_size=3
height=5
batch=3


inputs=np.random.randint(9,size=(batch,height,height,input_dim)).astype(np.float32)
# conv=tf.keras.layers.Conv2D(self.channels, (self.kernel_size, self.kernel_size), strides=(1, 1), input_shape=(self.height, self.width, 1), padding='same', activation='relu', kernel_initializer='uniform')
model = tf.keras.Sequential()
offital_conv=tf.keras.layers.Conv2D(channels, (kernel_size, kernel_size), strides=(1, 1), input_shape=(height, height, input_dim), padding='same', activation='relu', kernel_initializer='uniform')
own_conv=ConvLayer(input_dim, channels, kernel_size, 1)


print('offitial')
res1=offital_conv(inputs)
# print((offital_conv(inputs)))
print(res1.shape)
print('own')
res2=own_conv.forward(inputs.transpose(0,3,1,2))
print(res2.shape)
print()

subs=(res2.transpose(0,2,3,1)-res1)
print(np.sum(subs))

print(res1)

# Launch the graph in a session.
# sess = tf.Session()
# Evaluate the tensor `c`.
with tf.Session():
 # We can also use 'c.eval()' here.
    print(res1.eval())
print()
print()
print()
# print(res2.transpose(0,2,3,1))
