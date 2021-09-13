import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn

import argparse
import numpy as np
import pandas as pd

from resnet18 import ResNet_BN as ResNet
from solver import get_dataloader, train_loop, test_loop
from plot import plot_loss_and_acc

max_iter = 60e4
batch_size = 256
total_layer=18
epochs = int(max_iter // (1281167 // batch_size))

pngname = (
    "ddp_imagenet_aug_bnB_cross_MultiStepLR_layer"
    + str(total_layer)
    + "_epo_"
    + str(epochs)
    + "_batch_"
    + str(batch_size)
)

print(pngname)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',default=0,type=int,help='node rank for distributed training')
    parser.add_argument('--total_layer',default=total_layer,type=int,help='config layer of ResNet')
    parser.add_argument('--batch_size',default=batch_size,type=int,help='batch size')
    return parser.parse_args()


def main():
    
    args = get_arguments()
    print(args.local_rank)
    cudnn.benchmark = True

    torch.distributed.init_process_group(backend='nccl',init_method='env://')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    #load the dataset
    
    train_dataloader, test_dataloader = get_dataloader(batch_size=args.batch_size,sampler=True)

    net = ResNet(total_layer=args.total_layer).cuda()
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch=-1
    )
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
    data_records_pd = pd.DataFrame(data_records)
    data_records.to_csv("csv/" + filename + ".csv")
    print("Done!")

if __name__=='__main__':
    main()