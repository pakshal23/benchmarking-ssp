import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import scipy.io
import numpy as np


def init_dataset(params):
    """
    params:
        data_type: Type of data to be used (eg: 'stochastic_process')
        data_dir: Directory containing the data
        num_train: Number of training signals
        num_valid: Number of validation signals
        num_test: Number of test signals
    """
    
    # add your datasets here and create a corresponding Dataset class
    dataset_dict = {'stochastic_process' : SSPDataset}

    if params.data_type not in dataset_dict.keys():
        raise ValueError('Chosen dataset is not available... Exiting.')

    dataset = dataset_dict[params.data_type](params)

    return dataset


class SSPDataset():
    """Synthetic dataset of sparse stochastic processes (load .mat file)."""

    def __init__(self, params):
        """
        params:
            data_dir: Directory containing the data
            num_train: Number of training signals
            num_valid: Number of validation signals
            num_test: Number of test signals
        """
        self.params = params
        
        
    def get_dataset(self, dataset_type='train'):
        """ This function returns a tensordataset and the forward model
        """
        
        filename = self.params.data_dir + '_' + dataset_type + '.mat'
        data_mat = scipy.io.loadmat(filename)
        
        # Load the forward model
        H = torch.tensor(data_mat['H'], dtype=torch.float)
        M = H.shape[0]
        K = H.shape[1]

        if (self.params.exp_type == "fourier_sampling"):
            print("**** Fourier sampling experiment ****")
            H_temp = H.clone()
            H = torch.zeros([2*M-1, K])
            H[0,:] = H_temp[0,:]
            H[1:M,:] = H_temp[1:,:]
            H[M:(2*M-1),:] = H_temp[1:,:]
            H = H/K
    
        if (dataset_type == 'train'):
            num_samples = self.params.num_train
        elif (dataset_type == 'valid'):
            num_samples = self.params.num_valid
        elif (dataset_type == 'test'):
            num_samples = self.params.num_test
            
        x_data = torch.empty([K, num_samples], dtype=torch.float)
        X = data_mat['x_cell']
        Y = data_mat['y_cell']

        if (self.params.exp_type == "fourier_sampling"):
            y_data = torch.empty([2*M-1, num_samples], dtype=torch.float)
            for i in range(num_samples):
                x_data[:, i:i+1] = torch.tensor(X[0, i])
                tmp = torch.tensor(Y[0, i])
                y_data[0, i:i+1] = tmp[0]
                y_data[1:M,i:i+1] = tmp[1:M]
                y_data[M:2*M-1,i:i+1] = tmp[1:M]

            #y_data = torch.empty([M, num_samples], dtype=torch.float)
            #for i in range(num_samples):
            #    x_data[:, i:i+1] = torch.tensor(X[0, i])
            #    y_data[:, i:i+1] = torch.tensor(Y[0, i])
        else:
            y_data = torch.empty([M, num_samples], dtype=torch.float)
            for i in range(num_samples):
                x_data[:, i:i+1] = torch.tensor(X[0, i])
                y_data[:, i:i+1] = torch.tensor(Y[0, i])

        #y_data = torch.empty([M, num_samples], dtype=torch.float)
        #for i in range(num_samples):
        #    x_data[:, i:i+1] = torch.tensor(X[0, i])
        #    y_data[:, i:i+1] = torch.tensor(Y[0, i])
    
        x_data = torch.transpose(x_data, 0, 1)   # num_samples X K (length of signal)
        y_data = torch.transpose(y_data, 0, 1)   # num_samples X M (length of measurements) 
        
        data = TensorDataset(y_data, x_data)
    
    
        return (data, H)
        
