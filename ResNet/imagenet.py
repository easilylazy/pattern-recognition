# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import numpy as np
from resnet18 import ResNet_BN as ResNet

from torchvision import datasets
from torchvision import transforms
import torch
import torch.backends.cudnn as cudnn

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from plot import plot_loss_and_acc
import pandas as pd

# %%
# params
batch_size = 256
learning_rate = 0.1
total_layer = 18
max_iter = 64e4
epochs = int(max_iter // (50000 // batch_size))
own = True
own = False
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
device_ids = [0, 1]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=batch_size,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)

args = parser.parse_args()


pngname = (
    "dp_otherstd_aug_bnB_cross_MultiStepLR_layer"
    + str(total_layer)
    + "_delr_lr_"
    + str(learning_rate)
    + "_epo_"
    + str(epochs)
    + "_batch_"
    + str(batch_size)
)

print(pngname)


try:
    torch.cuda.empty_cache()
    print("success clean")
except:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = ResNet(total_layer=total_layer).cuda()
# %%

loss_fn = nn.CrossEntropyLoss().cuda()
# for learning_rate in range(0.01)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0.0001,momentum=0.9)
optimizer = optim.SGD(
    net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], last_epoch=-1
)


cudnn.benchmark = True
# %% data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_dataset():
    """
        Uses torchvision.datasets.ImageNet to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

    trainset = datasets.ImageNet(
        "/home/share/datasets/imagenet",
        split="train",
        transform=train_transforms,
        target_transform=None,
    )
    valset = datasets.ImageNet(
        "/home/share/datasets/imagenet",
        split="val",
        transform=val_transforms,
        target_transform=None,
    )
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss

        X, y = X.cuda(), y.cuda()

        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
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
            X, y = X.cuda(), y.cuda()
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
train_dataloader, test_dataloader = get_dataset()
print("data loaded")
test_loop(test_dataloader, net, loss_fn)
SAVE_CKP = False

avg_train_loss, avg_train_acc = [], []
avg_test_loss, avg_test_acc = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_acc = train_loop(train_dataloader, net, loss_fn, optimizer)
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

loss = np.max(avg_test_loss)
accu = np.max(avg_test_acc)

## plot
filename = pngname + "_acc_" + str(accu) + "_loss_" + str(loss)
plot_loss_and_acc(
    {"train": [avg_train_loss, avg_train_acc], "test": [avg_test_loss, avg_test_acc]},
    filename=filename,
)

## record
foo = {}
foo["test_loss"] = avg_test_loss
foo["test_acc"] = avg_test_acc
foo["train_loss"] = avg_train_loss
foo["train_acc"] = avg_train_acc
bar = pd.DataFrame(foo)
bar.to_csv("csv/" + filename + ".csv")
print("Done!")

