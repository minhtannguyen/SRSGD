import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

def mulvt(v,t):
##################################
## widely used in binary search ##
## v is batch_size by 1         ##
## t is batch_size by any dim   ##
##################################
    batch_size, other_dim = t.size()[0], t.size()[1:]
    len_dim = len(other_dim)-1
    for i in range(len_dim):
        v = v.unsqueeze(len(v.size()))
    v = v.expand(t.size())
    return v*t    
    
def reduce_sum(t,axis):
    dim = t.size()

class FGSM(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,xi,label_or_target,TARGETED):
        criterion = nn.CrossEntropyLoss()
        output = self.model.predict(xi)
        #print(output, label_or_target)
        loss = criterion(output, label_or_target)
        #print(loss)
        #print(c.size(),modifier.size())
        return loss

    def i_fgsm(self, input_xi, label_or_target, eta, bound, TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        x_adv = Variable(input_xi.cuda(), requires_grad=True)
        for it in range(20):
            error = self.get_loss(x_adv,yi,TARGETED)
            # print(error.data[0]) 
            self.model.get_gradient(error)
            #print(gradient)
            x_adv.grad.sign_()
            if TARGETED:
                x_adv.data = x_adv.data - eta* x_adv.grad 
                x_adv.data = torch.clamp(x_adv.data, bound[0], bound[1])
            else:
                x_adv.data = x_adv.data + eta* x_adv.grad
                x_adv.data = torch.clamp(x_adv.data, bound[0], bound[1])
            #x_adv = Variable(x_adv.data, requires_grad=True)
            #error.backward()
        return x_adv

    def fgsm(self, input_xi, label_or_target, eta, bound, TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        x_adv = Variable(input_xi.cuda(), requires_grad=True)

        error = self.get_loss(x_adv,yi,TARGETED)
        self.model.get_gradient(error)
        #print(gradient)
        x_adv.grad.sign_()
        if TARGETED:
            x_adv.data = x_adv.data - eta* x_adv.grad 
            x_adv.data = torch.clamp(x_adv.data, bound[0], bound[1])
        else:
            x_adv.data = x_adv.data + eta* x_adv.grad
            x_adv.data = torch.clamp(x_adv.data, bound[0], bound[1])
            #x_adv = Variable(x_adv.data, requires_grad=True)
            #error.backward()
        return x_adv 

    def __call__(self, input_xi, label_or_target, eta=0.01, TARGETED=False):
        adv = self.i_fgsm(input_xi, label_or_target, eta, TARGETED)
        return adv   


        