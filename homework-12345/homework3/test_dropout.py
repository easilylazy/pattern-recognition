import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from datetime import datetime

tf.disable_eager_execution()

from network import Network
from solver import train, test
from plot import plot_loss_and_acc
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
def decode_image(image):
    # Normalize from [0, 255.] to [0., 1.0], and then subtract by the mean value
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [1, 28, 28])
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
pngname='dropout_fc1'
batch_size = 50
max_epoch = 20
init_std = 0.01

learning_rate = 0.01
weight_decay = 0.01

disp_freq = 50

from criterion import SoftmaxCrossEntropyLossLayer
from optimizer import SGD

criterion = SoftmaxCrossEntropyLossLayer()
sgd = SGD(learning_rate, weight_decay)


from layers import FCLayer, ReLULayer, ConvLayer, MaxPoolingLayer, ReshapeLayer, DropoutLayer

convNet = Network()

size=7


convNet.add(ConvLayer(1, 8, 3, 1))
convNet.add(ReLULayer())
convNet.add(MaxPoolingLayer(2, 0))
convNet.add(ConvLayer(8, 16, 3, 1))
convNet.add(ReLULayer())
convNet.add(MaxPoolingLayer(2, 0))
convNet.add(ReshapeLayer((batch_size, 16, 7, 7), (batch_size, 784)))
convNet.add(FCLayer(16* size* size, 128))
convNet.add(DropoutLayer())
convNet.add(ReLULayer())
convNet.add(FCLayer(128, 10))

# Train
convNet.is_training = True

start=datetime.now()
convNet, conv_loss, conv_acc = train(convNet, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)
cost_time=datetime.now() - start

print('sigmoid euclidean time: '+str(cost_time.microseconds))

pngname +='_time_'+str(cost_time.microseconds)

# Test
convNet.is_training = False
accu=test(convNet, criterion, data_test, batch_size, disp_freq)
# accu=np.max(conv_acc)
loss=np.min(conv_loss)
filename = (pngname+'_lr_'+str(learning_rate) + "_de_"
    + str(weight_decay)
    + "_bat_"
    + str(batch_size)+'_epo_'+str(max_epoch)+'_acc_'+str(accu)+'_loss_'+str(loss)
    )
plot_loss_and_acc({"dropout": [conv_loss, conv_acc]}, filename=filename)
import pandas as pd
foo={}
foo['train_loss']=conv_loss
foo['train_acc']=conv_acc
bar=pd.DataFrame(foo)
bar.to_csv('csv/'+filename+'.csv')
print('done!')