# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from pickle import FALSE
from input_data import DataSets
import input_data
import matplotlib.pyplot as plt
import numpy as np
from net import ResNet_BN as ResNet
# from resnet import resnet20 as ResNet
import torchvision
from torchvision import transforms
import torch
import torch.backends.cudnn as cudnn




# %%
# params
batch_size=128
learning_rate = 0.1
unit_num=6
max_iter = 64e3
epochs = 200#int(max_iter//(50000//batch_size))
own=True
own=False
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5,8,9"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
device_ids = [0, 1]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                    metavar='N', help='mini-batch size (default: 128)')

args = parser.parse_args()


if own:
    pngname = "dp_ownstd_aug_bnB_cross_sched_layer"+str(6*(unit_num+1)+2)+"_delr_lr_"+str(learning_rate)+'_epo_'+str(epochs)+'_batch_'+str(batch_size)
else:
    pngname = "dp_otherstd_aug_bnB_cross_MultiStepLR_layer"+str(6*(unit_num+1)+2)+"_delr_lr_"+str(learning_rate)+'_epo_'+str(epochs)+'_batch_'+str(batch_size)

print(pngname)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.distributed.init_process_group(backend="nccl")

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = ResNet(unit_num=unit_num).cuda()
try:
    torch.cuda.empty_cache()  # PyTorch thing
    print("success clean")
except:
    pass
# %% data
from torch.utils.data import DataLoader

if own:
    mnist_dataset = input_data.load_data(path='data/cifar-10-batches-py',one_hot=False,augment=True)

    train_dataloader = DataLoader(mnist_dataset.train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_dataset.test, batch_size=batch_size, shuffle=False)
else:
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),

    cudnn.benchmark = True

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])

    train_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss

        X,y=X.cuda(),y.cuda()

        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        correct += (pred.data.argmax(1) == y).type(torch.int).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    correct /= size
    return train_loss, correct

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X,y=X.cuda(),y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss, correct

# %%
import torch.optim as optim

loss_fn = nn.CrossEntropyLoss().cuda()
# for learning_rate in range(0.01)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0.0001,momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=-1)
# %%
test_loop(test_dataloader, net, loss_fn)
SAVE_CKP = False

avg_train_loss, avg_train_acc = [], []
avg_test_loss, avg_test_acc = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss,train_acc=train_loop(train_dataloader, net, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, net, loss_fn)
    avg_train_loss.append(train_loss)
    avg_train_acc.append(train_acc)
    avg_test_loss.append(test_loss)
    avg_test_acc.append(test_acc)
    filename = "checkpoint\model_w_epoch" + pngname + str(t) + ".pth"
    if SAVE_CKP:
        torch.save(net.state_dict(), filename)
        print("save in " + filename)
    # if t==epochs//2 or t==epochs//4*3:
    #     learning_rate*=0.1
    #     optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0.0001,momentum=0.9)
    scheduler.step()

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

