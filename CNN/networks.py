import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parameter import Parameter
from torch import Tensor
import time


############################################# Simple CNN with residual connection ##################################################
def conv_unit(num_channels, filter_size):
    return nn.Sequential(nn.Conv1d(num_channels, num_channels, filter_size, groups = 1, padding = int((filter_size-1)/2), bias=False), 
                         nn.BatchNorm1d(num_channels, momentum=0.1, affine = True), 
                         nn.ReLU()
           )


class ResCNN(nn.Module):

    def __init__(self, params, device):
        """ """
        super().__init__()

        self.params = params
        self.device = device
                
        self.filter_size = self.params.filter_size
        self.num_layers = self.params.num_layers
        self.num_channels = self.params.num_channels
        
        conv_units = [conv_unit(self.num_channels, self.filter_size) for i in range(self.num_layers-2)]
        
        self.network = nn.Sequential(nn.Conv1d(1, self.num_channels, self.filter_size, padding = int((self.filter_size-1)/2), bias=False), 
                       nn.BatchNorm1d(self.num_channels, momentum=0.1, affine = True),              
                       nn.ReLU(), 
                       *conv_units, 
                       nn.Conv1d(self.num_channels, 1, self.filter_size, padding = int((self.filter_size-1)/2))
                       )

        
        
    def get_num_params(self):
        """ """
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params

    

    def parameters_conv(self):
        """ """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                for param in module.parameters():
                    yield param
                
                
                
    def parameters_bn(self):
        """ """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                for param in module.parameters():
                    yield param
                

    
    def set_forward_model(self, H):
        
        self.H = H.to(self.device)
        self.N = H.shape[1]
        self.M = H.shape[0]
        #self.HTH = torch.matmul(torch.t(self.H), self.H)


    
    def get_initial_reconstruction(self, y_batch):
        """ y_batch: batch_size X M
        """
        bs = y_batch.shape[0]
        yb = torch.reshape(y_batch,[bs, 1, self.M])
        HTy = torch.matmul(yb, self.H)
        
        return torch.reshape(HTy, [bs, self.N])
        
        
        
    def forward(self, y_batch):
        
        bs = y_batch.shape[0]
        yb = torch.reshape(y_batch,[bs, 1, self.M])
        HTy = torch.matmul(yb, self.H)   
        
        #t_s = time.time()
        sig = HTy + self.network(HTy)
        #total_time = (time.time()-t_s)
                       
        #return torch.reshape(sig, [bs, self.N]), total_time
        return torch.reshape(sig, [bs, self.N])



############################################# Simple CNN with residual connection but no batchnorm ##################################################
def conv_unit_no_bn(num_channels, filter_size):
    return nn.Sequential(nn.Conv1d(num_channels, num_channels, filter_size, groups = 1, padding = int((filter_size-1)/2), bias=True), 
                         nn.ReLU()
           )


class ResCNN_no_bn(nn.Module):

    def __init__(self, params, device):
        """ """
        super().__init__()

        self.params = params
        self.device = device
                
        self.filter_size = self.params.filter_size
        self.num_layers = self.params.num_layers
        self.num_channels = self.params.num_channels
        
        conv_units = [conv_unit_no_bn(self.num_channels, self.filter_size) for i in range(self.num_layers-2)]
        
        self.network = nn.Sequential(nn.Conv1d(1, self.num_channels, self.filter_size, padding = int((self.filter_size-1)/2), bias=True), 
                       nn.ReLU(), 
                       *conv_units, 
                       nn.Conv1d(self.num_channels, 1, self.filter_size, padding = int((self.filter_size-1)/2))
                       )

        
        
    def get_num_params(self):
        """ """
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params

    

    def parameters_conv(self):
        """ """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                for param in module.parameters():
                    yield param
                
                
                
    def parameters_bn(self):
        """ """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                for param in module.parameters():
                    yield param
                

    
    def set_forward_model(self, H):
        
        self.H = H.to(self.device)
        self.N = H.shape[1]
        self.M = H.shape[0]
        #self.HTH = torch.matmul(torch.t(self.H), self.H)


    
    def get_initial_reconstruction(self, y_batch):
        """ y_batch: batch_size X M
        """
        bs = y_batch.shape[0]
        yb = torch.reshape(y_batch,[bs, 1, self.M])
        HTy = torch.matmul(yb, self.H)
        
        return torch.reshape(HTy, [bs, self.N])
        
        
        
    def forward(self, y_batch):
        
        bs = y_batch.shape[0]
        yb = torch.reshape(y_batch,[bs, 1, self.M])
        HTy = torch.matmul(yb, self.H)   
        
        sig = HTy + self.network(HTy)
                       
        return torch.reshape(sig, [bs, self.N])