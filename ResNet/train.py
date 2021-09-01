# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from pickle import FALSE
import input_data
import matplotlib.pyplot as plt
import numpy as np


# %%
mnist_dataset = input_data.load_data(path='data/cifar-10-batches-py',one_hot=True)


# %%
# params
pngname = "resnet_lr_"
drop_out=0.3
epochs = 10
batch_size=128
learning_rate = 0.1

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(mnist_dataset.train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(mnist_dataset.test, batch_size=batch_size, shuffle=True)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1_0 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv2_0 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3_0 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)

        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,10)  
        self.fc_dropout = nn.Dropout(drop_out) 

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x1_0 = F.relu(self.conv1_0(x))
        x = F.relu(self.conv1(x1_0))
        x = F.relu(self.conv1(x))

        x2_0=x1_0+x
        x = F.relu(self.conv2_0(x2_0))
        x2_1 = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x2_1))
        x = F.relu(self.conv2(x))

        x3_0=x2_1+x
        x = F.relu(self.conv3_0(x3_0))
        x = F.relu(self.conv3(x))

        # print(x.shape)

        x = self.pool(x)
        # print(x.shape)
        # x = F.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x


net = Net().to(device)
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
        X = X.reshape(X.shape[0], 3, 32, 32).to(torch.float32).to(device)
        pred = model(X)
        loss = loss_fn(pred, y.to(device).float())

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
            X = X.reshape(X.shape[0], 3, 32, 32).to(torch.float32).to(device)
            y = y.to(device)
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
# for learning_rate in range(0.01)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0.0001,momentum=0.9)

# %%
model_load = net.to(device)
try:
    model_load.load_state_dict(torch.load("model_weights.pth"))
    model_load.eval()
except:
    pass

# %%
test_loop(test_dataloader, model_load, loss_fn)
SAVE_CKP = False
avg_test_loss, avg_test_acc = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model_load, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model_load, loss_fn)
    avg_test_loss.append(test_loss)
    avg_test_acc.append(test_acc)
    filename = "checkpoint\model_w_epoch" + pngname + str(t) + ".pth"
    if SAVE_CKP:
        torch.save(model_load.state_dict(), filename)
        print("save in " + filename)
loss,accu=test_loop(test_dataloader, model_load, loss_fn)
from plot import plot_loss_and_acc

filename = pngname+str(learning_rate)+'_epo_'+str(epochs)+'_acc_'+str(accu)+'_loss_'+str(loss)
plot_loss_and_acc({"ConvNet": [avg_test_loss, avg_test_acc]}, filename=filename)

print("Done!")
