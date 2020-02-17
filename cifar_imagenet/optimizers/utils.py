"""
Some helper functions for PyTorch.
"""
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

def init_params(net):
    """
    Initial layer parameters.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = time.time()

def progress_bar(current, total, msg=None):
    """
    The progress bar.
    """
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))
    
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def freeze_layer(layer):
    """
    Freeze a certain layer in the DNN.
    #Argument: the name of a layer in the given DNN.
    """
    for param in layer.parameters():
        param.requires_grad = False


def freeze_All(model):
    """
    Freeze all the trainable parameters in the DNN.
    #Argument: the DNN model.
    """
    for param in model.parameters(): # Parameter is a method of nn.Module
        param.requires_grad = False


def Unfreeze_layer(layer):
    """
    Unfreeze a given layer in t he DNN.
    #Argument: the name of a layer in the given DNN.
    """
    for param in layer.parameters():
        param.requires_grad = True

def Unfreeze_All(model):
    """
    Unfreeze a given layer in t he DNN.
    #Argument: DNN model.
    """
    for param in model.parameters():
        param.requires_grad = True
