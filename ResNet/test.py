from resnet18 import ResNet_BN as ResNet
from torch import optim
import torch
from collections import OrderedDict
import time

new_state_dict = OrderedDict()

gpu=9 # which gpu to use
loc = 'cuda:{}'.format(gpu)
# which checkpoint to load
resume='checkpoint/ddp_imagenet_aug_bnB_cross_MultiStepLR_layer18_epo_149_batch_320checkpoint.pth.tar'
data_dir="/home/share/datasets/imagenet"
total_layer=18
net = ResNet(total_layer=total_layer).cuda(gpu)
optimizer = optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

checkpoint = torch.load(resume, map_location=loc)
start_epoch = checkpoint['epoch']
best_acc1 = checkpoint['best_acc1']
test=checkpoint['state_dict']
for k,v in test.items():
    name=k[7:]
    new_state_dict[name]=v
net.load_state_dict(new_state_dict)
optimizer.load_state_dict(checkpoint['optimizer'])
print("=> loaded checkpoint '{}' (epoch {})"
        .format(resume, checkpoint['epoch']))
from torchvision import datasets
from torchvision import transforms
import os
valdir = os.path.join(data_dir, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)
print('data loaded')
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def validate(val_loader, model, criterion,gpu=0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    print_freq=1

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
validate(val_loader, net, criterion,gpu)