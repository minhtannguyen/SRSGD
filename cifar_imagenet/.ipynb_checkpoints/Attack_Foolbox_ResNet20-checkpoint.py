# -*- coding: utf-8 -*-
"""
CW, FGSM, and IFGSM Attack CNN
"""
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy
import math
import numpy as np
import os
import argparse

import torch.utils.data as data

# from utils import *

import numpy.matlib
import matplotlib.pyplot as plt
import pickle
# import cPickle
from collections import OrderedDict

import models.cifar as models

# import foolbox

from attack_toolbox import FGSM

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Fool EnResNet')
ap = parser.add_argument
ap('--method', help='Attack Method', type=str, default="ifgsm") # fgsm, ifgsm, cwl2
ap('--epsilon', help='Attack Strength', type=float, default=0.031) # May 2
ap('--num-ensembles', '--ne', default=2, type=int, metavar='N')
ap('--noise-coef', '--nc', default=0.1, type=float, metavar='W', help='forward noise (default: 0.0)')
ap('--noise-coef-eval', '--nce', default=0.0, type=float, metavar='W', help='forward noise (default: 0.)')

ap('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
ap('--depth', type=int, default=29, help='Model depth.')
ap('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
ap('--cardinality', type=int, default=8, help='Model cardinality (group).')
ap('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
ap('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
ap('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
ap('--feature_vec', default='x', type=str)
ap('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
ap('-d', '--dataset', default='cifar10', type=str)
ap('--eta', default=1.0, type=float, help='eta in HOResNet')
ap('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

opt = parser.parse_args()


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PytorchModel(object):
    def __init__(self,model, bounds, num_classes):
        self.model = model
        self.model.eval()
        self.bounds = bounds
        self.num_classes = num_classes
        self.num_queries = 0
    
    def predict(self,image):
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        output = self.model(image)
        self.num_queries += 1
        return output
 
    def predict_prob(self,image):
        with torch.no_grad():
            image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
            if len(image.size())!=4:
                image = image.unsqueeze(0)
            output = self.model(image)
            self.num_queries += 1
        return output
    
    def predict_label(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        with torch.no_grad():
            output = self.model(image)
            self.num_queries += image.size(0)
        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_ensemble(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            output.zero_()
            for i in range(10):
                output += self.model(image)
                self.num_queries += image.size(0)

        _, predict = torch.max(output.data, 1)
        
        return predict[0]

    def get_num_queries(self):
        return self.num_queries

    def get_gradient(self,loss):
        loss.backward()

if __name__ == '__main__':
    """
    Load the trained DNN, and attack the DNN, finally save the adversarial images
    """
    # Model
    if opt.dataset == 'cifar10':
        dataloader = dset.CIFAR10
        num_classes = 10
    else:
        dataloader = dset.CIFAR100
        num_classes = 100
        
    print("==> creating model '{}'".format(opt.arch))
    if opt.arch.startswith('resnext'):
        net = models.__dict__[opt.arch](
                    cardinality=opt.cardinality,
                    num_classes=num_classes,
                    depth=opt.depth,
                    widen_factor=opt.widen_factor,
                    dropRate=opt.drop,
                )
    elif opt.arch.startswith('densenet'):
        net = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                    growthRate=opt.growthRate,
                    compressionRate=opt.compressionRate,
                    dropRate=opt.drop,
                )
    elif opt.arch.startswith('wrn'):
        net = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                    widen_factor=opt.widen_factor,
                    dropRate=opt.drop,
                )
    elif opt.arch.startswith('resnet'):
        net = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                    block_name=opt.block_name,
                )
    elif opt.arch.startswith('preresnet'):
        net = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                    block_name=opt.block_name,
                )
    elif opt.arch.startswith('horesnet'):
        net = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                    eta=opt.eta,
                    block_name=opt.block_name,
                    feature_vec=opt.feature_vec
                )
    elif opt.arch.startswith('hopreresnet'):
        net = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                    eta=opt.eta,
                    block_name=opt.block_name,
                    feature_vec=opt.feature_vec
                )
    else:
        net = models.__dict__[opt.arch](num_classes=num_classes)
    
#     net = torch.nn.DataParallel(net).cuda()
#     cudnn.benchmark = True
    # Load the model
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(opt.checkpoint), 'Error: no checkpoint directory found!'
    opt.checkpoint_dir = os.path.dirname(opt.checkpoint)
    checkpoint = torch.load(opt.checkpoint)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    
    epsilon = opt.epsilon
    attack_type = opt.method
    
    # fnet = foolbox.models.PyTorchModel(net, bounds=(0, 1), num_classes=10)
    
    fnet = PytorchModel(net, bounds=(0,1), num_classes=10)
    adv_box = FGSM(fnet)
    
    if attack_type == 'fgsm':
        attack = adv_box.fgsm
    elif attack_type == 'ifgsm':
        attack = adv_box.i_fgsm
    else:
        print("Please choose the attacks provided")
        
    
    # Load the original test data
    print('==> Load the clean image')
    root = './data'
    download = False
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = 100
    if attack_type == 'cw':
        batchsize_test = 1
    print('Batch size of the test set: ', batchsize_test)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_set = dataloader(root='./data', train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(test_set, batch_size=batchsize_test, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    #--------------------------------------------------------------------------
    # Testing
    # images: the original images
    # labels: labels of the original images
    # images_adv: adversarial image
    # labels_pred: the predicted labels of the adversarial images
    # noise: the added noise
    #--------------------------------------------------------------------------
    images, labels, images_adv, labels_pred, noise = [], [], [], [], []
    total_fooled = 0; total_correct_classified = 0
    
    for batch_idx, (x1, y1_true) in enumerate(test_loader):
        #if batch_idx < 2:
        # x_Test = x1.numpy()
        #print x_Test.min(), x_Test.max()
        #x_Test = ((x_Test - x_Test.min())/(x_Test.max() - x_Test.min()) - 0.5)*2
        #x_Test = (x_Test - x_Test.min() )/(x_Test.max() - x_Test.min())
        # y_Test = y1_true.numpy()
        
        x_adversarial = attack(x1, y1_true, eta=epsilon, bound=(0,1))
        pred_tmp = net(x_adversarial)
        y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy(), axis=1)
        y_Test = y1_true.cpu().data.numpy()
        x_Test = x1.cpu().data.numpy()
            
        for i in range(len(y_Test)):
            #print y_pred_adversarial
            if y_Test[i] == y_pred_adversarial[i]:
                #if y_Test == y_pred_adversarial:
                total_correct_classified += 1
            
        for i in range(len(y_Test)):
            # Save the perturbed data
            images.append(x_Test[i, :, :, :]) # Original image
            images_adv.append(x_adversarial.detach().cpu().numpy()[i, :, :, :]) # Perturbed image
            noise.append(x_adversarial.detach().cpu().numpy()[i, :, :, :]-x_Test[i, :, :, :]) # Noise
            labels.append(y_Test[i])
            labels_pred.append(y_pred_adversarial[i])
    
    print(opt.checkpoint)
    print('Number of correctly classified images: ', total_correct_classified)
    # Save data
    #with open("Adversarial" + attack_type + str(int(10*epsilon)) + ".pkl", "w") as f:
    #with open("Adversarial" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
    #    adv_data_dict = {"images":images_adv, "labels":labels}
    #    cPickle.dump(adv_data_dict, f)
    images = np.array(images).squeeze()
    images_adv = np.array(images_adv).squeeze()
    noise = np.array(noise).squeeze()
    labels = np.array(labels).squeeze()
    labels_pred = np.array(labels_pred).squeeze()
    print([images.shape, images_adv.shape, noise.shape, labels.shape, labels_pred.shape])

#     with open("fooled_EnResNet5_20_PGD_10iters_" + attack_type + str(int(1000*epsilon)) + ".pkl", "w") as f:
#     #with open("fooled_EnResNet5_20_PGD_20iters_" + attack_type + str(int(1000*epsilon)) + ".pkl", "w") as f:
#         adv_data_dict = {
#             "images" : images,
#             "images_adversarial" : images_adv,
#             "y_trues" : labels,
#             "noises" : noise,
#             "y_preds_adversarial" : labels_pred
#             }
#         pickle.dump(adv_data_dict, f)
