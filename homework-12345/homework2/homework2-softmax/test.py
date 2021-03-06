import os, sys
import numpy as np
from softmax_classifier import softmax_classifier
import mnist_data_loader
import matplotlib.pyplot as plt

mnist_dataset = mnist_data_loader.read_data_sets("../MNIST_data/", one_hot=True)

# training dataset
train_set = mnist_dataset.train
# test dataset
test_set = mnist_dataset.test

train_size = train_set.num_examples
test_size = test_set.num_examples
print()
print("Training dataset size: ", train_size)
print("Test dataset size: ", test_size)

batch_size = 20
max_epoch = 20
learning_rate = 0.05

# For regularization
lamda = 0  # .02


# Weight Initialization
W = np.random.randn(28 * 28, 10) * 0.001

loss_set = []
accu_set = []
disp_freq = 100

# Training process
for epoch in range(0, max_epoch):
    iter_per_batch = train_size // batch_size
    for batch_id in range(0, iter_per_batch):
        batch = train_set.next_batch(batch_size)  # get data of next batch
        input, label = batch

        # softmax_classifier
        loss, gradient, prediction = softmax_classifier(W, input, label, lamda)

        # Calculate accuracy
        label = np.argmax(label, axis=1)  # scalar representation
        accuracy = sum(prediction.reshape(batch_size) == label) / float(len(label))

        loss_set.append(loss)
        accu_set.append(accuracy)

        # Update weights
        W = W - (learning_rate * gradient)
        if batch_id % disp_freq == 0:
            print(
                "Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                    epoch, max_epoch, batch_id, iter_per_batch, loss[0], accuracy
                )
            )
    print()
correct = 0
iter_per_batch = test_size // batch_size

# Test process
for batch_id in range(0, iter_per_batch):
    batch = test_set.next_batch(batch_size)
    data, label = batch

    # We only need prediction results in testing
    _, _, prediction = softmax_classifier(W, data, label, lamda)
    label = np.argmax(label, axis=1)

    correct += sum(prediction.reshape(batch_size) == label)

accuracy = correct * 1.0 / test_size
print("Test Accuracy: ", accuracy)


# training loss curve
path = "res/"
abbr = (
    "la_"
    + str(lamda)
    + "_lr_"
    + str(learning_rate)
    + "_acc_"
    + str(accuracy)
    + "_epo_"
    + str(max_epoch)
    + "_bat_"
    + str(batch_size)
)
plt.figure()
plt.plot(loss_set, "b--")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig(path + "loss_" + abbr + ".png")
# training accuracy curve
plt.figure()
plt.plot(accu_set, "r--")
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.savefig(path + "acc_" + abbr + ".png")
