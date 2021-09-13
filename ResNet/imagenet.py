# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from plot import plot_loss_and_acc
from solver import get_dataloader, train_loop, test_loop
from resnet18 import ResNet_BN as ResNet

# %%
# params
batch_size = 256
learning_rate = 0.1
total_layer = 18
max_iter = 60e4
epochs = int(max_iter // (50000 // batch_size))
own = True
own = False
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
device_ids = [0]


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



# %%
train_dataloader, test_dataloader = get_dataloader(batch_size)
print('data loaded')
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
data_records = {}
data_records["test_loss"] = avg_test_loss
data_records["test_acc"] = avg_test_acc
data_records["train_loss"] = avg_train_loss
data_records["train_acc"] = avg_train_acc
bar = pd.DataFrame(data_records)
bar.to_csv("csv/" + filename + ".csv")
print("Done!")

