# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from pickle import FALSE
import input_data
import matplotlib.pyplot as plt
import numpy as np


# %%
mnist_dataset = input_data.read_data_sets("..\homework-2\MNIST_data", one_hot=True)


# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(mnist_dataset.train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(mnist_dataset.test, batch_size=64, shuffle=True)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 7 * 7, 128)  # 7*7 from image dimension
        # self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
try:
    torch.cuda.empty_cache()  # PyTorch thing
    print("success clean")
except:
    pass


# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.reshape(X.shape[0], 1, 28, 28)
        pred = model(X)
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.reshape(X.shape[0], 1, 28, 28)
            pred = model(X)
            test_loss += loss_fn(pred, y.float()).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss, correct


# %%
import torch.optim as optim

loss_fn = nn.MSELoss()
# nn.CrossEntropyLoss()
learning_rate = 0.1
# for learning_rate in range(0.01)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# %%
model_load = net
try:
    model_load.load_state_dict(torch.load("model_weights.pth"))
    model_load.eval()
except:
    pass

# %%
test_loop(test_dataloader, model_load, loss_fn)
SAVE_CKP = False
epochs = 20
avg_test_loss, avg_test_acc = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model_load, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model_load, loss_fn)
    avg_test_loss.append(test_loss)
    avg_test_acc.append(test_acc)
    filename = "checkpoint\model_w_epoch" + str(t) + ".pth"
    if SAVE_CKP:
        torch.save(model_load.state_dict(), filename)
        print("save in " + filename)
_,accu=test_loop(test_dataloader, model_load, loss_fn)
from plot import plot_loss_and_acc

filename = "convnet2fc_lr_" + str(learning_rate)+'_acc_'+str(accu)
plot_loss_and_acc({"ConvNet": [avg_test_loss, avg_test_acc]}, filename=filename)

print("Done!")
