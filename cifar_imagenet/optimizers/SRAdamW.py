# -*- coding: utf-8 -*-
"""
Scheduled restarting adam with warm up
"""
import math
import torch
from .optimizer import Optimizer, required

class SRAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4, use_variance=True, warmup=1000, iter_count=1, restarting_iter=50):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_variance=use_variance, warmup=warmup, iter_count=iter_count, restarting_iter=restarting_iter)
        print('======== Warmup: {} ========='.format(warmup))
        super(SRAdamW, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SRAdamW, self).__setstate__(state)
    
    def update_iter(self):
        idx = 1
        for group in self.param_groups:
            if idx == 1:
                group['iter_count'] += 1
                if group['iter_count'] >= group['restarting_iter']:
                    group['iter_count'] = 1
            idx += 1
        return group['iter_count'], group['restarting_iter']
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = (group['iter_count'] - 1.)/(group['iter_count'] + 2.)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
                p_data_fp32 = p.data.float()
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32) # Tag
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32) # Tag
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                exp_avg.mul_(momentum).add_(1-momentum, grad) # Tag
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                #bias_correction1 = 1 - beta1 ** state['step'] # momentum ** state['step'] # Tag
                bias_correction1 = 1 - momentum ** state['step'] # Tag
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-6 + state['step'] * (group['lr'] - 1e-6) / group['warmup']
                else:
                    scheduled_lr = group['lr']
                
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(1., grad)
                grad = grad.add(momentum, buf)
                
                step_size = scheduled_lr * math.sqrt(bias_correction2)/bias_correction1 # double check if we need this bias_correction1
                
                if weight_decay != 0:
                    p_data_fp32.add_(-weight_decay*scheduled_lr, p_data_fp32)
                p_data_fp32.addcdiv_(-step_size, grad, denom)
                p.data.copy_(p_data_fp32)
        return loss


'''
class SRAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, use_variance=True, warmup=1000, iter_count=1, restarting_iter=50):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_variance=use_variance, warmup=warmup, iter_count=iter_count, restarting_iter=restarting_iter)
        print('======== Warmup: {} ========='.format(warmup))
        super(SRAdamW, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SRAdamW, self).__setstate__(state)
    
    def update_iter(self):
        idx = 1
        for group in self.param_groups:
            if idx == 1:
                group['iter_count'] += 1
                if group['iter_count'] >= group['restarting_iter']:
                    group['iter_count'] = 1
            idx += 1
        return group['iter_count'], group['restarting_iter']
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = (group['iter_count'] - 1.)/(group['iter_count'] + 2.)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data.float()
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)
                
                p_data_fp32 = p.data.float()
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-6 + state['step'] * (group['lr'] - 1e-6) / group['warmup']
                else:
                    scheduled_lr = group['lr']
                
                
        return loss
 '''                   


'''
class SRAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, use_variance=True, warmup = 1000, iter_count=1, restarting_iter=50):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_variance=True, warmup = warmup, iter_count=iter_count, restarting_iter=restarting_iter)
        print('======== Warmup: {} ========='.format(warmup))
        super(SRAdamW, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SRAdamW, self).__setstate__(state)
    
    def update_iter(self):
        idx = 1
        for group in self.param_groups:
            if idx == 1:
                group['iter_count'] += 1
                if group['iter_count'] >= group['restarting_iter']:
                    group['iter_count'] = 1
            idx += 1
        return group['iter_count'], group['restarting_iter']
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            # Compute the momentum parameter
            momentum = (group['iter_count'] - 1.)/(group['iter_count'] + 2.)
            # Weight decay parameter, i.e., l2 regularization
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data # gradient
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                param_state = self.state[p]
                
                if 'momentum_buffer' not in param_state:
                    buf0 = param_state['momentum_buffer'] = torch.clone(p.data).detach()
                else:
                    buf0 = param_state['momentum_buffer']
                
                buf1 = p.data - 
                
                
                
                
                p_data_fp32 = p.data.float()
                
                
                
                
                
        return loss
'''