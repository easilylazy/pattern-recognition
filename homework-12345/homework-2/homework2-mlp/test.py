import numpy as np 
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from network import Network
from solver import train, test
from plot import plot_loss_and_acc

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
def decode_image(image):
    # Normalize from [0, 255.] to [0., 1.0], and then subtract by the mean value
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    image = image / 255.0
    image = image - tf.reduce_mean(image)
    return image

def decode_label(label):
    # Encode label with one-hot encoding
    return tf.one_hot(label, depth=10)
    # Data Preprocessing
x_train = tf.data.Dataset.from_tensor_slices(x_train).map(decode_image)
y_train = tf.data.Dataset.from_tensor_slices(y_train).map(decode_label)
data_train = tf.data.Dataset.zip((x_train, y_train))

x_test = tf.data.Dataset.from_tensor_slices(x_test).map(decode_image)
y_test = tf.data.Dataset.from_tensor_slices(y_test).map(decode_label)
data_test = tf.data.Dataset.zip((x_test, y_test))
batch_size = 100
max_epoch = 20
init_std = 0.01

learning_rate_SGD = 0.001
weight_decay = 0.1

disp_freq = 50

from criterion import EuclideanLossLayer
from criterion import SoftmaxCrossEntropyLossLayer
# from tf.losses import softmax_cross_entropy
# criterion = tf.losses.softmax_cross_entropy()
criterion = SoftmaxCrossEntropyLossLayer()
from optimizer import SGD

# criterion = EuclideanLossLayer()

sgd = SGD(learning_rate_SGD, weight_decay)

from layers import FCLayer, SigmoidLayer

# sigmoidMLP = Network()
# # Build MLP with FCLayer and SigmoidLayer
# # 128 is the number of hidden units, you can change by your own
# sigmoidMLP.add(FCLayer(784, 128))
# sigmoidMLP.add(SigmoidLayer())
# sigmoidMLP.add(FCLayer(128, 10))

# sigmoidMLP, sigmoid_loss, sigmoid_acc = train(sigmoidMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)
# test(sigmoidMLP, criterion, data_test, batch_size, disp_freq)
# plot_loss_and_acc({'Sigmoid': [sigmoid_loss, sigmoid_acc],
#                    })

from layers import ReLULayer

reluMLP = Network()
# TODO build ReLUMLP with FCLayer and ReLULayer
reluMLP.add(FCLayer(784, 128))
# reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))

reluMLP, relu_loss, relu_acc = train(reluMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)

test(reluMLP, criterion, data_test, batch_size, disp_freq)

plot_loss_and_acc({                   'relu': [relu_loss, relu_acc]},title='test_sys_soft')