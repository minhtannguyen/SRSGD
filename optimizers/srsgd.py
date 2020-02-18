import torch
from .optimizer import Optimizer, required

class SRSGD(Optimizer):
    """
    Stochastic gradient descent with Adaptively restarting (200 iters) Nesterov momentum.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        iter_count (integer): count the iterations mod 200
    Example:
         >>> optimizer = torch.optim.SRSGD(model.parameters(), lr=0.1, weight_decay=5e-4, iter_count=1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        >>> iter_count = optimizer.update_iter()
    Formula:
        v_{t+1} = p_t - lr*g_t
        p_{t+1} = v_{t+1} + (iter_count)/(iter_count+3)*(v_{t+1} - v_t)
    """
    def __init__(self, params, lr=required, weight_decay=0., iter_count=1, restarting_iter=100):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if iter_count < 1:
            raise ValueError("Invalid iter count: {}".format(iter_count))
        if restarting_iter < 1:
            raise ValueError("Invalid iter total: {}".format(restarting_iter))
        
        defaults = dict(lr=lr, weight_decay=weight_decay, iter_count=iter_count, restarting_iter=restarting_iter)
        super(SRSGD, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SRSGD, self).__setstate__(state)
    
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
        """
        Perform a single optimization step.
        Arguments: closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = (group['iter_count'] - 1.)/(group['iter_count'] + 2.)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay !=0:
                    d_p.add_(weight_decay, p.data)
                
                param_state = self.state[p]
                
                if 'momentum_buffer' not in param_state:
                    buf0 = param_state['momentum_buffer'] = torch.clone(p.data).detach()
                else:
                    buf0 = param_state['momentum_buffer']
                
                buf1 = p.data - group['lr']*d_p
                p.data = buf1 + momentum*(buf1 - buf0)
                param_state['momentum_buffer'] = buf1
        return loss
