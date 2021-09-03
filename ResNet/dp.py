# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from pickle import FALSE
from input_data import DataSets
import input_data
import matplotlib.pyplot as plt
import numpy as np
from net import ResNet_BN as ResNet
import torchvision
from torchvision import transforms
import torch


# %%
mnist_dataset = input_data.load_data(path='data/cifar-10-batches-py',one_hot=True,augment=True)


# %%
# params
batch_size=128
learning_rate = 0.1
unit_num=2
max_iter = 64e3
epochs = int(max_iter//(50000//batch_size))
pngname = "dp_otherdata_bn_cross_unit_layer"+str(6*(unit_num+1)+2)+"_delr_lr_"+str(learning_rate)+'_epo_'+str(epochs)+'_batch_'+str(batch_size)
print(pngname)

# %%
from torch.utils.data import DataLoader

# train_dataloader = DataLoader(mnist_dataset.train, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(mnist_dataset.test, batch_size=batch_size, shuffle=True)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_dataloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.distributed.init_process_group(backend="nccl")

device_ids = [0, 1]
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = ResNet(unit_num=unit_num).to(device)
try:
    torch.cuda.empty_cache()  # PyTorch thing
    print("success clean")
except:
    pass


# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # X = X.reshape(X.shape[0], 3, 32, 32).to(torch.float32).to(device)
        X = X.reshape(X.shape[0], 3, 32, 32).to(device)
        y = y.to(device)

        pred = model(X)
        # loss = loss_fn(pred, y.to(device))
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.cpu().detach()
        correct += (pred.cpu().detach().argmax(1) == y.cpu()).type(torch.float).sum().item()


        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    correct /= size
    return train_loss.numpy(), correct

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # X = X.reshape(X.shape[0], 3, 32, 32).to(torch.float32).to(device)
            X = X.reshape(X.shape[0], 3, 32, 32).to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss, correct


# %%
import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()
# for learning_rate in range(0.01)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0.0001,momentum=0.9)
# %%
model_load = net.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 就这一行
    model_load = nn.DataParallel(model_load, device_ids=device_ids)
try:
    model_load.load_state_dict(torch.load("model_weights.pth"))
    model_load.eval()
except:
    pass

# %%
test_loop(test_dataloader, model_load, loss_fn)
SAVE_CKP = False

avg_train_loss, avg_train_acc = [], []
avg_test_loss, avg_test_acc = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss,train_acc=train_loop(train_dataloader, model_load, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model_load, loss_fn)
    avg_train_loss.append(train_loss)
    avg_train_acc.append(train_acc)
    avg_test_loss.append(test_loss)
    avg_test_acc.append(test_acc)
    filename = "checkpoint\model_w_epoch" + pngname + str(t) + ".pth"
    if SAVE_CKP:
        torch.save(model_load.state_dict(), filename)
        print("save in " + filename)
    if t==epochs//2 or t==epochs//4*3:
        learning_rate*=0.1
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0.0001,momentum=0.9)
loss=np.max(avg_test_loss)
accu=np.max(avg_test_acc)

from plot import plot_loss_and_acc

filename = pngname+'_acc_'+str(accu)+'_loss_'+str(loss)
plot_loss_and_acc({"train": [avg_train_loss, avg_train_acc],"test": [avg_test_loss, avg_test_acc]}, filename=filename)
import pandas as pd
foo={}
foo['test_loss']=avg_test_loss
foo['test_acc']=avg_test_acc
foo['train_loss']=avg_train_loss
foo['train_acc']=avg_train_acc
bar=pd.DataFrame(foo)
bar.to_csv('csv/'+filename+'.csv')
print("Done!")

