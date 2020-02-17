'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import fcntl

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet, Bottleneck

from utils import Bar, Logger, AverageMeter, mkdir_p, savefig, LoggerDistributed
from utils.radam import RAdam, AdamW
from utils.lsradam import LSRAdam, LSAdamW

from optimizers.sgd_adaptive3 import *
from optimizers.SRAdamW import *
from optimizers.SRRAdam import *

from tensorboardX import SummaryWriter

# for loading LMDB
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import to_python_float
import io
from PIL import Image
try:
    import lmdb
except:
    pass
import torch.distributed as dist

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names + ['densenet264'] + ['resnet200']

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer sgd|adam|radam')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='beta2 for adam')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--restart-schedule', type=int, nargs='+', default=[80, 200, 500, 1000],
                        help='Restart at after these amounts of epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--model_name', default='sgd')

# DALI
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')

parser.add_argument('--local_rank', type=int, default=0,
                    help='rank of process')

# LSAdam
parser.add_argument('--sigma', default=0.1, type=float, help='sigma in LSAdam')

args = parser.parse_args()

# Set up DDP.
args.distributed = True
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.world_size = torch.distributed.get_world_size()

state = {k: v for k, v in args._get_kwargs()}

# logger
if args.local_rank == 0:
    if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)
    writer = SummaryWriter(os.path.join(args.checkpoint, 'tensorboard')) # write to tensorboard

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
cudnn.benchmark = True
torch.manual_seed(args.manualSeed)
cudnn.enabled = True
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# Subroutines for lmdb_loader
def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')

def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None,
        loader=lmdb_loader):
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)
    return data_set

best_top1 = 0  # best test top1 accuracy
best_top5 = 0  # best test top5 accuracy
batch_time_global = AverageMeter()
data_time_global = AverageMeter()

def main():
    global best_top1, best_top5
    
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code    
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    
    train_data = imagenet_lmdb_dataset(traindir, transform=train_transform)
    valid_data = imagenet_lmdb_dataset(validdir, transform=val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch,
        shuffle=(train_sampler is None),
        pin_memory=True, num_workers=8, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.test_batch, shuffle=False,
        pin_memory=True, num_workers=8)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    elif args.arch == 'densenet264':
        model = DenseNet(growth_rate=32, block_config=(6, 12, 64, 48),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False)
    elif args.arch == 'resnet200':
        model = ResNet(block=Bottleneck, layers=[3, 24, 36, 3], num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = DDP(model.features)
        model.cuda()
    else:
        model = model.cuda()
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, warmup = 0)
    elif args.optimizer.lower() == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lsadam': 
        optimizer = LSAdamW(model.parameters(), lr=args.lr*((1.+4.*args.sigma)**(0.25)), 
                           betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay, 
                           sigma=args.sigma)
    elif args.optimizer.lower() == 'lsradam':
        sigma = 0.1
        optimizer = LSRAdam(model.parameters(), lr=args.lr*((1.+4.*args.sigma)**(0.25)), 
                           betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay, 
                           sigma=args.sigma)
    elif args.optimizer.lower() == 'srsgd':
        iter_count = 1
        optimizer = SGD_Adaptive(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, iter_count=iter_count, restarting_iter=args.restart_schedule[0])
    elif args.optimizer.lower() == 'sradam':
        iter_count = 1
        optimizer = SRNAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, restarting_iter=args.restart_schedule[0]) 
    elif args.optimizer.lower() == 'sradamw':
        iter_count = 1
        optimizer = SRAdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = 0, restarting_iter=args.restart_schedule[0]) 
    elif args.optimizer.lower() == 'srradam':
        #NOTE: need to double-check this
        iter_count = 1
        optimizer = SRRAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = 0, restarting_iter=args.restart_schedule[0]) 
    
    schedule_index = 1
    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.checkpoint = os.path.dirname(args.resume)
        # checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.local_rank))
        checkpoint = torch.load(args.resume, map_location = torch.device('cpu'))
        best_top1 = checkpoint['best_top1']
        best_top5 = checkpoint['best_top5']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.optimizer.lower() == 'srsgd' or args.optimizer.lower() == 'sradam' or args.optimizer.lower() == 'sradamw' or args.optimizer.lower() == 'srradam':
            iter_count = optimizer.param_groups[0]['iter_count']
        schedule_index = checkpoint['schedule_index']
        state['lr'] =  optimizer.param_groups[0]['lr']
        if args.checkpoint == args.resume:
            logger = LoggerDistributed(os.path.join(args.checkpoint, 'log.txt'), rank=args.local_rank, title=title, resume=True)
        else:
            logger = LoggerDistributed(os.path.join(args.checkpoint, 'log.txt'), rank=args.local_rank, title=title)
            if args.local_rank == 0:
                logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Top1', 'Valid Top1', 'Train Top5', 'Valid Top5'])
    else:
        logger = LoggerDistributed(os.path.join(args.checkpoint, 'log.txt'), rank=args.local_rank, title=title)
        if args.local_rank == 0:
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Top1', 'Valid Top1', 'Train Top5', 'Valid Top5'])
    
    if args.local_rank == 0:
        logger.file.write('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    if args.evaluate:
        if args.local_rank == 0:
            logger.file.write('\nEvaluation only')
        test_loss, test_top1, test_top5 = test(val_loader, model, criterion, start_epoch, use_cuda, logger)
        if args.local_rank == 0:
            logger.file.write(' Test Loss:  %.8f, Test Top1:  %.2f, Test Top5: %.2f' % (test_loss, test_top1, test_top5))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        # Shuffle the sampler.
        train_loader.sampler.set_epoch(epoch + args.manualSeed)
        
        if args.optimizer.lower() == 'srsgd':    
            if epoch in args.schedule:
                current_lr = args.lr * (args.gamma**schedule_index)
                current_restarting_iter = args.restart_schedule[schedule_index]
                optimizer = SGD_Adaptive(model.parameters(), lr=current_lr, weight_decay=args.weight_decay, iter_count=iter_count, restarting_iter=current_restarting_iter)
                schedule_index += 1
            
            if epoch >= args.schedule[-1]:
                prev_schedule_index = schedule_index - 1
                current_lr = args.lr * (args.gamma**prev_schedule_index)
                start_decay_restarting_iter = args.restart_schedule[prev_schedule_index] - 1
                current_restarting_iter = start_decay_restarting_iter * (args.epochs - epoch - 1)/(args.epochs - args.schedule[-1] - 1) + 1
                optimizer = SGD_Adaptive(model.parameters(), lr=current_lr, weight_decay=args.weight_decay, iter_count=iter_count, restarting_iter=current_restarting_iter)
                
        else:
            adjust_learning_rate(optimizer, epoch)
        
        if args.local_rank == 0:
            logger.file.write('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        
        if args.optimizer.lower() == 'srsgd' or args.optimizer.lower() == 'sradam' or args.optimizer.lower() == 'sradamw' or args.optimizer.lower() == 'srradam':
            train_loss, train_top1, train_top5, iter_count = train(train_loader, model, criterion, optimizer, epoch, use_cuda, logger)
        else:
            train_loss, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch, use_cuda, logger)

        test_loss, test_top1, test_top5 = test(val_loader, model, criterion, epoch, use_cuda, logger)

        # append logger file
        if args.local_rank == 0:
            logger.append([state['lr'], train_loss, test_loss, train_top1, test_top1, train_top5, test_top5])
            writer.add_scalars('train_loss', {args.model_name: train_loss}, epoch)
            writer.add_scalars('test_loss', {args.model_name: test_loss}, epoch)
            writer.add_scalars('train_top1', {args.model_name: train_top1}, epoch)
            writer.add_scalars('test_top1', {args.model_name: test_top1}, epoch)
            writer.add_scalars('train_top5', {args.model_name: train_top5}, epoch)
            writer.add_scalars('test_top5', {args.model_name: test_top5}, epoch)

        # save model
        is_best = test_top1 > best_top1
        best_top1 = max(test_top1, best_top1)
        best_top5 = max(test_top5, best_top5)
        if args.local_rank == 0:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'schedule_index': schedule_index,
                    'state_dict': model.state_dict(),
                    'top1': test_top1,
                    'top5': test_top5,
                    'best_top1': best_top1,
                    'best_top5': best_top5,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, epoch, checkpoint=args.checkpoint)
            
            if epoch == args.schedule[-1]:
                logger.file.write('Best top1: %f at epoch %i'%(best_top1, epoch))
                logger.file.write('Best top5: %f at epoch %i'%(best_top5, epoch))
                print('Best top1: %f at epoch %i'%(best_top1, epoch))
                print('Best top5: %f at epoch %i'%(best_top5, epoch))
                with open("./all_results_imagenet.txt", "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write("%s\n"%args.checkpoint)
                    f.write("best_top1 %f, best_top5 %f at epoch %i\n\n"%(best_top1,best_top5,epoch))
                    fcntl.flock(f, fcntl.LOCK_UN)
                
    if args.local_rank == 0:
        logger.file.write('Best top1: %f'%best_top1)
        logger.file.write('Best top5: %f'%best_top5)
        logger.close()
        logger.plot()
        savefig(os.path.join(args.checkpoint, 'log.eps'))
        print('Best top1: %f'%best_top1)
        print('Best top5: %f'%best_top5)
        with open("./all_results_imagenet.txt", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write("%s\n"%args.checkpoint)
            f.write("best_top1 %f, best_top5 %f\n\n"%(best_top1,best_top5))
            fcntl.flock(f, fcntl.LOCK_UN)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda, logger):
    global batch_time_global, data_time_global
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    train_loader_len = len(train_loader)
    # print('Length of train loader = %i\n'%train_loader_len)
    bar = Bar('Processing', max=train_loader_len)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time_lap = time.time() - end
        data_time.update(data_time_lap)
        if epoch > 0:
            data_time_global.update(data_time_lap)
        
        n = inputs.size(0)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        # print('input size = %i, device %s\n'%(inputs.size(0), inputs.device))
        # compute output
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and step.
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        reduced_loss = reduce_tensor(loss.data, args.world_size)
        prec1 = reduce_tensor(prec1, args.world_size)
        prec5 = reduce_tensor(prec5, args.world_size)
        
        losses.update(to_python_float(reduced_loss), n)
        top1.update(to_python_float(prec1), n)
        top5.update(to_python_float(prec5), n)
        
        # for restarting
        if args.optimizer.lower() == 'srsgd' or args.optimizer.lower() == 'sradam' or args.optimizer.lower() == 'sradamw' or args.optimizer.lower() == 'srradam':
            iter_count, iter_total = optimizer.update_iter()

        # measure elapsed time
        batch_time_lap = time.time() - end
        batch_time.update(batch_time_lap)
        if epoch > 0:
            batch_time_global.update(batch_time_lap)
        end = time.time()

        # plot progress
        bar.suffix  = '(Epoch {epoch}, {batch}/{size}) Data: {data:.3f}s/{data_global:.3f}s | Batch: {bt:.3f}s/{bt_global:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=train_loader_len,
                    data=data_time.val,
                    data_global=data_time_global.avg,
                    bt=batch_time.val,
                    bt_global=batch_time_global.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
        if args.local_rank == 0:
            logger.file.write(bar.suffix)
    bar.finish()
    if args.optimizer.lower() == 'srsgd' or args.optimizer.lower() == 'sradam' or args.optimizer.lower() == 'sradamw' or args.optimizer.lower() == 'srradam':
        return (losses.avg, top1.avg, top5.avg, iter_count)
    else:
        return (losses.avg, top1.avg, top5.avg)

def test(val_loader, model, criterion, epoch, use_cuda, logger):
    global best_top1, best_top5

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    val_loader_len = len(val_loader)
    bar = Bar('Processing', max=val_loader_len)
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        n=inputs.size(0)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '(Epoch {epoch}, {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=val_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        if args.local_rank == 0:
            logger.file.write(bar.suffix)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    next_epoch = epoch + 1
    next_two_epoch = epoch + 2
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    if next_epoch in args.schedule:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_epoch_%i.pth.tar'%epoch))
    if next_two_epoch in args.schedule:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_epoch_%i.pth.tar'%epoch))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

if __name__ == '__main__':
    main()
    # writer.close()
