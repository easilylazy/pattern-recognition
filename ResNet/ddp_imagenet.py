import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import argparse
import numpy as np
import pandas as pd
import os 
import shutil


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
device = torch.device('cuda', 0)
def save_checkpoint(state, is_best, filename=pngname+'checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename=pngname+'model_best.pth.tar')

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',default=0,type=int,help='node rank for distributed training')
    parser.add_argument('--total_layer',default=total_layer,type=int,help='config layer of ResNet')
    parser.add_argument('--batch_size',default=batch_size,type=int,help='batch size')
    return parser.parse_args()
args = get_arguments()
def tradition():
    print(args.local_rank)
    cudnn.benchmark = True

    torch.distributed.init_process_group(backend='nccl',init_method='env://')
    torch.cuda.set_device(args.local_rank)
    #load the dataset
    
    train_dataloader, test_dataloader = get_dataloader(batch_size=args.batch_size,sampler=True)

    net = ResNet(total_layer=args.total_layer).cuda()
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 90], last_epoch=-1
    )
    print('data loaded')
    test_loop(test_dataloader, net, loss_fn, rank=args.local_rank)
    SAVE_CKP = False

    avg_train_loss, avg_train_acc = [], []
    avg_test_loss, avg_test_acc = [], []
    best_acc=0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc = train_loop(train_dataloader, net, loss_fn, optimizer, rank=args.local_rank)
        test_loss, test_acc = test_loop(test_dataloader, net, loss_fn, rank=args.local_rank)
        avg_train_loss.append(train_loss)
        avg_train_acc.append(train_acc)
        avg_test_loss.append(test_loss)
        avg_test_acc.append(test_acc)
        if best_acc<test_acc:
            is_best=True
            best_acc=test_acc
        else:
            is_best=False
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc1': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
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
    data_records_pd.to_csv("csv/" + filename + ".csv")
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12377'
    # os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()


def ddp_basic(rank, world_size):
    cudnn.benchmark = True

    print(f"Running basic DDP task on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    train_dataloader, test_dataloader = get_dataloader(batch_size=args.batch_size,sampler=True)
    print('data loaded')

    net = ResNet(total_layer=args.total_layer).to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank], find_unused_parameters=True)


    loss_fn = nn.CrossEntropyLoss().to(rank)

    optimizer = optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30,60], last_epoch=-1
    )
    test_loop(test_dataloader, net, loss_fn, rank=rank)
    best_acc=0.0
    avg_train_loss, avg_train_acc = [], []
    avg_test_loss, avg_test_acc = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc = train_loop(train_dataloader, net, loss_fn, optimizer, rank=rank)
        test_loss, test_acc = test_loop(test_dataloader, net, loss_fn, rank=rank)
        avg_train_loss.append(train_loss)
        avg_train_acc.append(train_acc)
        avg_test_loss.append(test_loss)
        avg_test_acc.append(test_acc)
        if best_acc<test_acc:
            is_best=True
            best_acc=test_acc
        else:
            is_best=False
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc1': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        scheduler.step()
    cleanup()
    
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
    data_records_pd.to_csv("csv/" + filename + ".csv")

def run_task(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__ == "__main__":
    print(pngname)
    # n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    # world_size = n_gpus
    # run_task(ddp_basic, world_size)

    tradition()
    
    print("Done!")