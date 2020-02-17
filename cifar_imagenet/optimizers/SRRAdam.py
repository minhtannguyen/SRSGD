# -*- coding: utf-8 -*-
"""
Scheduled restarting RAdam
"""
import math
import torch
from .optimizer import Optimizer, required

class SRRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4, iter_count=1, restarting_iter=50):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, iter_count=iter_count, restarting_iter=restarting_iter)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(SRRAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SRRAdam, self).__setstate__(state)
    
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
            #momentum = 0.9 # Test this
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('SRRAdam does not support sparse gradients')
                
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
                
                exp_avg_sq.mul_(beta2).addcmul_(1-beta2, grad, grad)
                exp_avg.mul_(momentum).add_(1-momentum, grad)
                
                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1-beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    
                    # More conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1-beta2_t)*(N_sma-4) / (N_sma_max-4)*(N_sma-2) / N_sma * N_sma_max / (N_sma_max-2)) / (1 - momentum**state['step'])
                    else:
                        step_size = group['lr'] / (1 - momentum ** state['step'])
                    
                    '''
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1-beta2_t)*(N_sma-4) / (N_sma_max-4)*(N_sma-2) / N_sma * N_sma_max / (N_sma_max-2)) / (1 - beta1**state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    '''
                    buffered[2] = step_size
                
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(1., grad)
                grad = grad.add(momentum, buf)
                
                if weight_decay != 0:
                    p_data_fp32.add_(-weight_decay*group['lr'], p_data_fp32)
                
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, grad, denom) # TODO: grad or exp_avg?
                else:
                    p_data_fp32.add_(-step_size, grad)
                
                p.data.copy_(p_data_fp32)
        
        return loss