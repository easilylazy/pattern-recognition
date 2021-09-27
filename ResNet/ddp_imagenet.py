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
import random
from collections import OrderedDict

from torch.utils.data import sampler


from imagenet import ResNet_BN as ResNet
from solver import get_dataloader, train_loop, test_loop
from plot import plot_loss_and_acc

max_iter = 60e4
batch_size = 256
total_layer = 18
epochs = int(max_iter // (1281167 // batch_size))
load_checkpoint = False
resume = "checkpoint/epo20.pth.tar"
# ddp_imagenet_aug_bnB_cross_MultiStepLR_layer18_epo_149_batch_320checkpoint.pth.tar'
os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "8"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,8,9"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
pngname = (
    "ddp_imagenet_inner_bnB_cross_MultiStepLR_layer"
    + str(total_layer)
    + "_epo_"
    + str(epochs)
    + "_batch_"
    + str(batch_size)
)

torch.cuda.empty_cache()  # PyTorch thing

def init_seeds(seed=0):
    random.seed(seed)
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, filename=pngname + "checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, pngname + "model_best.pth.tar")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total_layer", default=total_layer, type=int, help="config layer of ResNet"
    )
    parser.add_argument("--batch_size", default=batch_size, type=int, help="batch size")
    return parser.parse_args()


args = get_arguments()

def solver(rank, world_size):
    print(f"Running basic DDP task on rank {rank}.")
    setup(rank, world_size)
    # load the dataset

    cudnn.benchmark = True

    train_dataloader, test_dataloader, sampler = get_dataloader(
        batch_size=args.batch_size, sampler=True
    )

    net = ResNet(total_layer=args.total_layer).cuda(rank)
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda()

    loss_fn = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    print("data loaded")
    best_acc = 0.0
    if load_checkpoint:
        new_state_dict = OrderedDict()
        loc = "cuda:{}".format(rank)
        checkpoint = torch.load(resume, map_location=loc)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc1"]
        test = checkpoint["state_dict"]
        for k, v in test.items():
            name = k[7:]
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint["epoch"])
        )
    else:
        start_epoch = 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 90], last_epoch=start_epoch - 1
    )
    net = nn.parallel.DistributedDataParallel(
        net, device_ids=[rank], find_unused_parameters=True
    )

    avg_train_loss, avg_train_acc = [], []
    avg_test_loss, avg_test_acc = [], []
    for epoch in range(start_epoch, epochs):
        if rank == 0:
            print(f"Epoch {epoch}\n-------------------------------")
            print("lr: ", scheduler.get_last_lr())
        if sampler != None:
            sampler.set_epoch(epoch)
        train_loss, train_acc = train_loop(
            train_dataloader, net, loss_fn, optimizer, rank=rank
        )
        if rank == 0:
            test_loss, test_acc = test_loop(
                test_dataloader, net, loss_fn, rank=rank
            )
            avg_test_loss.append(test_loss)
            avg_test_acc.append(test_acc)
            if best_acc < test_acc:
                is_best = True
                best_acc = test_acc
            else:
                is_best = False
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_acc1": best_acc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )


        avg_train_loss.append(train_loss)
        avg_train_acc.append(train_acc)


        scheduler.step()
    if rank == 0:
        loss = np.max(avg_test_loss)
        accu = np.max(avg_test_acc)
        ## plot
        filename = pngname + "_acc_" + str(accu) + "_loss_" + str(loss)
        plot_loss_and_acc(
            {
                "train": [avg_train_loss, avg_train_acc],
                "test": [avg_test_loss, avg_test_acc],
            },
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
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12399"
    # os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def ddp_basic(rank, world_size):
    print(f"Running basic DDP task on rank {rank}.")
    setup(rank, world_size)



def run_task(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    print(pngname)
    init_seeds()
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    print('use ',world_size,' gpus')
    run_task(solver, world_size)

    print("Done!")
