#------------------------------------------------------------------------------
# System module.
#------------------------------------------------------------------------------
import os
import random
import time
import copy
import argparse
import sys

#------------------------------------------------------------------------------
# Torch module.
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

#------------------------------------------------------------------------------
# Numpy module.
#------------------------------------------------------------------------------
import numpy as np
import numpy.matlib

#------------------------------------------------------------------------------
# DNN module
#------------------------------------------------------------------------------
from resnet import *
from vgg import *
from utils import *

#from LS_SGD import *
from sgd import *
#from sgd_2 import *
#from sgd_3 import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    """
    The main function.
    """
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    
    #--------------------------------------------------------------------------
    # Load the Cifar10 data.
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = '../data_Cifar10'
    download = True
    
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    train_set = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    
    batchsize_test = len(test_set)/20
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    batchsize_train = 128
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    
    batchsize_train = len(train_set)
    print('Total training (known) batch number: ', len(train_loader))
    print('Total testing batch number: ', len(test_loader))
    
    #--------------------------------------------------------------------------
    # Build the model
    #--------------------------------------------------------------------------
    #net = resnet20().cuda()
    #net = resnet32().cuda()
    #net = resnet44().cuda()
    #net = resnet56().cuda()
    net = resnet110().cuda()
    
    #net = preact_resnet20().cuda()
    #net = preact_resnet32().cuda()
    #net = preact_resnet44().cuda()
    #net = preact_resnet56().cuda()
    
    #net = vgg11_bn().cuda()
    #net = vgg13_bn().cuda()
    #net = vgg16_bn().cuda()
    
    '''
    # Print the model's information
    paramsList = list(net.parameters())
    kk = 0
    for ii in paramsList:
        l = 1
        print('The structure of this layer: ' + str(list(ii.size())))
        for jj in ii.size():
            l *= jj
        print('The number of parameters in this layer: ' + str(l))
        kk = kk+l
    print('Total number of parameters: ' + str(kk))
    '''
    
    sigma = 0.5
    lr = 0.1 # Changed
    weight_decay = 5e-4
    
    #global iter_count
    iter_count = 1
    
    #optimizer = LS_SGD(net.parameters(), lr=lr, sigma = sigma, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    #optimizer = SGD2(net.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay)
    #optimizer = SGD3(net.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay, iter_count=iter_count)
    
    nepoch = 240 #200
    for epoch in xrange(nepoch):
        print('Epoch ID: ', epoch)
        
        #----------------------------------------------------------------------
        # Training
        #----------------------------------------------------------------------
        ''''
        if epoch >= 40 and (epoch//40 == epoch/40.0):
            lr = lr/10
            print("Descrease the Learning Rate, lr = ", lr)
            optimizer = LS_SGD(net.parameters(), lr=lr, sigma = sigma, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        '''
        
        if epoch == 80:
            optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay, nesterov=True)
            #optimizer = SGD2(net.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
            #optimizer = SGD3(net.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay, iter_count=iter_count)
        elif epoch == 120:
            optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay, nesterov=True)
            #optimizer = SGD2(net.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)
            #optimizer = SGD3(net.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay, iter_count=iter_count)
        elif epoch == 160:
            optimizer = SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay, nesterov=True)
            #optimizer = SGD2(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay)
            #optimizer = SGD3(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay, iter_count=iter_count)
        elif epoch == 160:
            optimizer = SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=weight_decay, nesterov=True)
            #optimizer = SGD2(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay)
            #optimizer = SGD3(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay, iter_count=iter_count)
        
        correct = 0; total = 0; train_loss = 0
        net.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score, loss = net(x, target)
            loss.backward()
            optimizer.step()
            #iter_tmp = optimizer.update_iter() # Only for adaptive momentum (k-1)/(k+2)
            #print('Iter: ', iter_tmp)
            train_loss += loss.item()#data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------
        test_loss = 0; correct = 0; total = 0
        net.eval()
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            score, loss = net(x, target)
            
            test_loss += loss.item() #data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving model...')
            state = {
                'net': net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint_Cifar'):
                os.mkdir('checkpoint_Cifar')
            torch.save(state, './checkpoint_Cifar/ckpt.t7')
            best_acc = acc
    
    print('The best acc: ', best_acc)
