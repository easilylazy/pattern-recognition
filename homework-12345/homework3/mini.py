import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from network import Network
from solver import train, test
from plot import plot_loss_and_acc
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
def decode_image(image):
    # Normalize from [0, 255.] to [0., 1.0], and then subtract by the mean value
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [1, 28, 28])
    # image = image / 255.0
    image = image - tf.reduce_mean(image)
    return image

def decode_label(label):
    # Encode label with one-hot encoding
    # return tf.one_hot(label, depth=10)
    return (np.arange(10)==label[:,None]).astype(np.integer)
# Data Preprocessing


batch_size = 2
max_epoch = 10
init_std = 0.01

learning_rate = 0.001
weight_decay = 0.005

disp_freq = 1

from criterion import SoftmaxCrossEntropyLossLayer
from optimizer import SGD

criterion = SoftmaxCrossEntropyLossLayer()
sgd = SGD(learning_rate, weight_decay)


from layers import FCLayer, ReLULayer, ConvLayer, MaxPoolingLayer, ReshapeLayer

convNet = Network()
convNet.add(ConvLayer(1, 8, 3, 1))
convNet.add(ReLULayer())
convNet.add(MaxPoolingLayer(2, 0))
convNet.add(ConvLayer(8, 16, 3, 1))
convNet.add(ReLULayer())
convNet.add(MaxPoolingLayer(2, 0))
convNet.add(ReshapeLayer((batch_size, 16, 7, 7), (batch_size, 784)))
convNet.add(FCLayer(784, 128))
convNet.add(ReLULayer())
convNet.add(FCLayer(128, 10))

# Train
convNet.is_training = True
# convNet, conv_loss, conv_acc = train(convNet, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)

def minitest(model, criterion, optimizer, disp_freq, epoch):
    train_x = (np.random.random((batch_size,1,28,28)))
    train_y = np.array([1,2])

    # Forward pass
    logit = model.forward(train_x)
    print('forward success')
    criterion.forward(logit, np.array(decode_label(train_y)))
    print('cri forward success')


    # Backward pass
    delta = criterion.backward()
    model.backward(delta)
    print('backward success')


    # Update weights, see optimize.py
    optimizer.step(model)
    print('cri backward success')


    # Record loss and accuracy
    batch_train_loss=(criterion.loss)
    batch_train_acc=(criterion.acc)

    # if iteration % disp_freq == 0:
    print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
            epoch, max_epoch, iteration, max_train_iteration,
            np.mean(batch_train_loss), np.mean(batch_train_acc)))

minitest(convNet, criterion, sgd, disp_freq, 0)
