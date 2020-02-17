# -*- coding: utf-8 -*-
"""
momentum net
"""
import torch
import torch.nn as nn
import math

from torch.nn.parameter import Parameter

__all__ = ['momentumnet_heavyball_learned_scalar']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, step_size=2.0, momentum=0.5):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        # for momentum net
        self.step_size = Parameter(torch.tensor(step_size), requires_grad=True)
        self.momentum = Parameter(torch.tensor(momentum), requires_grad=True)

    def forward(self, invec):
        x, y = invec[0], invec[1]
        
        residualx = x
        residualy = y

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residualx = self.downsample(x)
            residualy = self.downsample(y)
        
        outy = residualx - self.step_size*out
        outx = (1.0 + self.momentum) * outy - self.momentum * residualy
        
        return [outx, outy]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, step_size=2.0, momentum=0.5):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # for momentum net
        self.step_size = Parameter(torch.tensor(step_size), requires_grad=True)
        self.momentum = Parameter(torch.tensor(momentum), requires_grad=True)

    def forward(self, invec):
        x, prex = invec[0], invec[1]
        
        residualx = x
        residualprex = prex

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residualx = self.downsample(x)
            residualprex = self.downsample(prex)
        
        outx = residualx - self.step_size*out + self.momentum * (residualx - residualprex)
        
        return [outx, residualx]


class MomentumNet(nn.Module):

    def __init__(self, depth, step_size=2.0, momentum=0.5, num_classes=1000, block_name='BasicBlock', feature_vec='x'):
        super(MomentumNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        # for momentum net
        self.step_size = step_size
        self.momentum = momentum
        
        self.feature_vec = feature_vec
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, step_size=self.step_size, momentum=self.momentum)
        self.layer2 = self._make_layer(block, 32, n, stride=2, step_size=self.step_size, momentum=self.momentum)
        self.layer3 = self._make_layer(block, 64, n, stride=2, step_size=self.step_size, momentum=self.momentum)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, step_size=2.0, momentum=0.5):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, step_size=step_size, momentum=momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, step_size=step_size, momentum=momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        
        out = [x, x]
        out = self.layer1(out)  # 32x32
        out = self.layer2(out)  # 16x16
        out = self.layer3(out)  # 8x8
        
        if self.feature_vec=='x':
            x = out[0]
        else:
            x = out[1]
            
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def momentumnet_heavyball_learned_scalar(**kwargs):
    """
    Constructs a ResNet model.
    """
    return MomentumNet(**kwargs)

# def momentum_net20(**kwargs):
#     return MomentumNet(num_classes=10, depth=20, block_name="basicblock")

# def momentum_net56(**kwargs):
#     return MomentumNet(num_classes=10, depth=56, block_name="bottleneck")

# def momentum_net110(**kwargs):
#     return MomentumNet(num_classes=10, depth=110, block_name="bottleneck")

# def momentum_net164(**kwargs):
#     return MomentumNet(num_classes=10, depth=164, block_name="bottleneck")

# def momentum_net290(**kwargs):
#     return MomentumNet(num_classes=10, depth=290, block_name="bottleneck")
