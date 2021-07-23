# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
# from IPython import get_ipython

# %% [markdown]
# # Homework-2: MLP for MNIST Classification
# 
# ### In this homework, you need to
# - #### implement SGD optimizer (`./optimizer.py`)
# - #### implement forward and backward for FCLayer (`layers/fc_layer.py`)
# - #### implement forward and backward for SigmoidLayer (`layers/sigmoid_layer.py`)
# - #### implement forward and backward for ReLULayer (`layers/relu_layer.py`)
# - #### implement EuclideanLossLayer (`criterion/euclidean_loss.py`)
# - #### implement SoftmaxCrossEntropyLossLayer (`criterion/softmax_cross_entropy.py`)

# %%
from os import stat
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from network import Network
from solver import train, test
from plot import plot_loss_and_acc
# %%
from datetime import datetime

start=datetime.now()

a=3
for i in range(100):
    a+=1
cost_time=datetime.now() - start

print('time: '+str(cost_time.microseconds))

# %% [markdown]
# ## Load MNIST Dataset
# We use tensorflow tools to load dataset for convenience.

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# %%
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


# %%
# Data Preprocessing
x_train = tf.data.Dataset.from_tensor_slices(x_train).map(decode_image)
y_train = tf.data.Dataset.from_tensor_slices(y_train).map(decode_label)
data_train = tf.data.Dataset.zip((x_train, y_train))

x_test = tf.data.Dataset.from_tensor_slices(x_test).map(decode_image)
y_test = tf.data.Dataset.from_tensor_slices(y_test).map(decode_label)
data_test = tf.data.Dataset.zip((x_test, y_test))

# %% [markdown]
# ## Set Hyerparameters
# You can modify hyerparameters by yourself.

# %%
batch_size = 100
max_epoch = 0
init_std = 0.01

learning_rate_SGD = 0.001
weight_decay = 0.1

disp_freq = 50

test_choice = False

# %% [markdown]
# ## 1. MLP with Euclidean Loss
# In part-1, you need to train a MLP with **Euclidean Loss**.  
# **Sigmoid Activation Function** and **ReLU Activation Function** will be used respectively.
# ### TODO
# Before executing the following code, you should complete **./optimizer.py** and **criterion/euclidean_loss.py**.

# %%
from criterion import EuclideanLossLayer
from optimizer import SGD

criterion = EuclideanLossLayer()

sgd = SGD(learning_rate_SGD, weight_decay)

# %% [markdown]
# ## 1.1 MLP with Euclidean Loss and Sigmoid Activation Function
# Build and train a MLP contraining one hidden layer with 128 units using Sigmoid activation function and Euclidean loss function.
# 
# ### TODO
# Before executing the following code, you should complete **layers/fc_layer.py** and **layers/sigmoid_layer.py**.

# %%
from layers import FCLayer, SigmoidLayer

sigmoidMLP = Network()
# Build MLP with FCLayer and SigmoidLayer
# 128 is the number of hidden units, you can change by your own
sigmoidMLP.add(FCLayer(784, 128))
sigmoidMLP.add(SigmoidLayer())
sigmoidMLP.add(FCLayer(128, 10))


# %%
start=datetime.now()

sigmoidMLP, sigmoid_loss, sigmoid_acc = train(sigmoidMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)

cost_time=datetime.now() - start

print('sigmoid euclidean time: '+str(cost_time.microseconds))



# %%
if test_choice:
    test(sigmoidMLP, criterion, data_test, batch_size, disp_freq)

# %% [markdown]
# ## 1.2 MLP with Euclidean Loss and ReLU Activation Function
# Build and train a MLP contraining one hidden layer with 128 units using ReLU activation function and Euclidean loss function.
# 
# ### TODO
# Before executing the following code, you should complete **layers/relu_layer.py**.

# %%
from layers import ReLULayer

reluMLP = Network()
# TODO build ReLUMLP with FCLayer and ReLULayer
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))


# %%
start=datetime.now()

reluMLP, relu_loss, relu_acc = train(reluMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)

cost_time=datetime.now() - start

print('Relu euclidean time: '+str(cost_time.microseconds))



# %%
if test_choice:
    test(reluMLP, criterion, data_test, batch_size, disp_freq)

# %% [markdown]
# ## Plot

# %%
plot_loss_and_acc({'Sigmoid': [sigmoid_loss, sigmoid_acc],
                   'relu': [relu_loss, relu_acc]})

# %% [markdown]
# ## 2. MLP with Softmax Cross-Entropy Loss
# In part-2, you need to train a MLP with **Softmax Cross-Entropy Loss**.  
# **Sigmoid Activation Function** and **ReLU Activation Function** will be used respectively again.
# ### TODO
# Before executing the following code, you should complete **criterion/softmax_cross_entropy_loss.py**.

# %%
from criterion import SoftmaxCrossEntropyLossLayer

criterion = SoftmaxCrossEntropyLossLayer()

sgd = SGD(learning_rate_SGD, weight_decay)

# %% [markdown]
# ## 2.1 MLP with Softmax Cross-Entropy Loss and Sigmoid Activation Function
# Build and train a MLP contraining one hidden layer with 128 units using Sigmoid activation function and Softmax cross-entropy loss function.

# %%
sigmoidMLP = Network()
# Build MLP with FCLayer and SigmoidLayer
# 128 is the number of hidden units, you can change by your own
sigmoidMLP.add(FCLayer(784, 128))
sigmoidMLP.add(SigmoidLayer())
sigmoidMLP.add(FCLayer(128, 10))

# %% [markdown]
# ### Train

# %%
start=datetime.now()

sigmoidMLP, sigmoid_loss, sigmoid_acc = train(sigmoidMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)

cost_time=datetime.now() - start

print('sigmoid softmax time: '+str(cost_time.microseconds))


# %% [markdown]
# ### Test

# %%
if test_choice:

    test(sigmoidMLP, criterion, data_test, batch_size, disp_freq)

# %% [markdown]
# ## 2.2 MLP with Softmax Cross-Entropy Loss and ReLU Activation Function
# Build and train a MLP contraining one hidden layer with 128 units using ReLU activation function and Softmax cross-entropy loss function.

# %%
reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))


# %%
start=datetime.now()

reluMLP, relu_loss, relu_acc = train(reluMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)

cost_time=datetime.now() - start

print('Relu softmax time: '+str(cost_time.microseconds))


# %%
if test_choice:

    test(reluMLP, criterion, data_test, batch_size, disp_freq)

# %% [markdown]
# ## Plot

# %%
plot_loss_and_acc({'Sigmoid': [sigmoid_loss, sigmoid_acc],
                   'relu': [relu_loss, relu_acc]})

# %% [markdown]
# ### ~~You have finished homework2-mlp, congratulations!~~  
# 
# **Next, according to the requirements 4) of report:**
# ### **You need to construct a two-hidden-layer MLP, using any activation function and loss function.**
# 
# **Note: Please insert some new cells blow (using '+' bottom in the toolbar) refer to above codes. Do not modify the former code directly.**

