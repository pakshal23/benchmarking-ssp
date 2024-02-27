import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parameter import Parameter
from torch import Tensor


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
        self.HTH = torch.matmul(torch.t(self.H), self.H)


    
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
    
    
    
########################################### UNet architecture with residual connection #########################################
class ResUNet(nn.Module):
    
    def __init__(self, params, device):
        """ """
        super().__init__()

        self.params = params
        self.device = device
        
        self.filter_size = self.params.filter_size
        self.num_channels = self.params.num_channels
        
        # create network model
        self.block_1_1 = None
        self.block_2_1 = None
        self.block_3_1 = None
        self.block_4_1 = None
        self.block_5 = None
        self.block_4_2 = None
        self.block_3_2 = None
        self.block_2_2 = None
        self.block_1_2 = None
        self.create_model(self.filter_size, self.num_channels)

        

    def create_model(self, kernel_size, num_channels):
        padding = int((self.filter_size-1)/2)

        # block_1_1
        block_1_1 = []
        block_1_1.extend(self.add_block_conv(in_channels=1, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_1_1 = nn.Sequential(*block_1_1)

        # block_2_1
        block_2_1 = [nn.MaxPool1d(kernel_size=2)]
        block_2_1.extend(self.add_block_conv(in_channels=num_channels, out_channels=2*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_1.extend(self.add_block_conv(in_channels=2*num_channels, out_channels=2*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_2_1 = nn.Sequential(*block_2_1)

        # block_3_1
        block_3_1 = [nn.MaxPool1d(kernel_size=2)]
        block_3_1.extend(self.add_block_conv(in_channels=2*num_channels, out_channels=4*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_1.extend(self.add_block_conv(in_channels=4*num_channels, out_channels=4*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_3_1 = nn.Sequential(*block_3_1)

        # block_4_1
        block_4_1 = [nn.MaxPool1d(kernel_size=2)]
        block_4_1.extend(self.add_block_conv(in_channels=4*num_channels, out_channels=8*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_1.extend(self.add_block_conv(in_channels=8*num_channels, out_channels=8*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_4_1 = nn.Sequential(*block_4_1)

        # block_5
        block_5 = [nn.MaxPool1d(kernel_size=2)]
        block_5.extend(self.add_block_conv(in_channels=8*num_channels, out_channels=16*num_channels, kernel_size=kernel_size, stride=1,
                                           padding=padding, batchOn=True, ReluOn=True))
        block_5.extend(self.add_block_conv(in_channels=16*num_channels, out_channels=16*num_channels, kernel_size=kernel_size, stride=1,
                                           padding=padding, batchOn=True, ReluOn=True))
        block_5.extend(self.add_block_conv_transpose(in_channels=16*num_channels, out_channels=8*num_channels, kernel_size=kernel_size, stride=2,
                                                     padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_5 = nn.Sequential(*block_5)

        # block_4_2
        block_4_2 = []
        block_4_2.extend(self.add_block_conv(in_channels=16*num_channels, out_channels=8*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(self.add_block_conv(in_channels=8*num_channels, out_channels=8*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(
            self.add_block_conv_transpose(in_channels=8*num_channels, out_channels=4*num_channels, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_4_2 = nn.Sequential(*block_4_2)

        # block_3_2
        block_3_2 = []
        block_3_2.extend(self.add_block_conv(in_channels=8*num_channels, out_channels=4*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(self.add_block_conv(in_channels=4*num_channels, out_channels=4*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(
            self.add_block_conv_transpose(in_channels=4*num_channels, out_channels=2*num_channels, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_3_2 = nn.Sequential(*block_3_2)

        # block_2_2
        block_2_2 = []
        block_2_2.extend(self.add_block_conv(in_channels=4*num_channels, out_channels=2*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(self.add_block_conv(in_channels=2*num_channels, out_channels=2*num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(
            self.add_block_conv_transpose(in_channels=2*num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_2_2 = nn.Sequential(*block_2_2)

        # block_1_2
        block_1_2 = []
        block_1_2.extend(self.add_block_conv(in_channels=2*num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=num_channels, out_channels=1, kernel_size=1, stride=1,
                                             padding=0, batchOn=False, ReluOn=False))
        self.block_1_2 = nn.Sequential(*block_1_2)

        
        
    @staticmethod
    def add_block_conv(in_channels, out_channels, kernel_size, stride, padding, batchOn, ReluOn):
        seq = []
        # conv layer
        conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=False)
        seq.append(conv)

        # batch norm layer
        if batchOn:
            batch_norm = nn.BatchNorm1d(num_features=out_channels, affine=True)
            seq.append(batch_norm)

        # relu layer
        if ReluOn:
            seq.append(nn.ReLU())
        return seq

    
    
    @staticmethod
    def add_block_conv_transpose(in_channels, out_channels, kernel_size, stride, padding, output_padding, batchOn, ReluOn):
        seq = []

        convt = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, output_padding=output_padding, bias=False)
        seq.append(convt)

        if batchOn:
            batch_norm = nn.BatchNorm1d(num_features=out_channels, affine=True)
            seq.append(batch_norm)

        if ReluOn:
            seq.append(nn.ReLU())
        return seq
    
    
    
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
        self.HTH = torch.matmul(torch.t(self.H), self.H)


    
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
        
        block_1_1_output = self.block_1_1(HTy)
        block_2_1_output = self.block_2_1(block_1_1_output)
        block_3_1_output = self.block_3_1(block_2_1_output)
        block_4_1_output = self.block_4_1(block_3_1_output)
        block_5_output = self.block_5(block_4_1_output)
        result = self.block_4_2(torch.cat((block_4_1_output, block_5_output), dim=1))
        result = self.block_3_2(torch.cat((block_3_1_output, result), dim=1))
        result = self.block_2_2(torch.cat((block_2_1_output, result), dim=1))
        result = self.block_1_2(torch.cat((block_1_1_output, result), dim=1))
        result = result + HTy
                       
        return torch.reshape(result, [bs, self.N])
    
    
    
"""
    def named_parameters_conv(self, recurse=True):
        """ """
        for name, param in self.named_parameters(recurse=recurse):
            conv_param = False
            
            if (name.endswith('weight') and (param.dim > 1)):
                conv_param = True

            if conv_param is True:
                yield name, param



    def named_parameters_no_conv(self, prefix='', recurse=True):
        """ """
        for name, param in self.named_parameters(recurse=recurse):
            conv_param = False
            
            if (name.endswith('weight') and (param.dim > 1)):
                conv_param = True

            if conv_param is False:
                yield name, param



    def parameters_conv(self, recurse=True):
        """ """
        for name, param in self.named_parameters_conv(recurse=recurse):
            yield param



    def parameters_no_conv(self, recurse=True):
        """ """
        for name, param in self.named_parameters_no_conv(recurse=recurse):
            yield param
"""