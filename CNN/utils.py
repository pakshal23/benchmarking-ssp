import json
from easydict import EasyDict
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config
        
        except ValueError:
            print("INVALID JSON file format.")
            exit(-1)
            
            
            
def get_data_loader(dataset, params):
    """
    Generate a data_loader from the given dataset
    Params:
        dataset: dataset for training (Should be a PyTorch dataset)
        params:
            batch_size: batch size of the data
            num_workers: num of parallel readers
    """
    
    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers, drop_last=True, pin_memory=False
    )

    return data_loader  


def get_finite_diff_matrix(K):
    """
    Create the first-order finite-difference matrix (zero-boundary conditions)
    Params:
        K: size of the matrix (K x K)
    """
    L = torch.zeros([K, K], dtype=torch.float)
    for kk in range(K-1):
        L[kk+1, kk] = -1.0
        L[kk+1, kk+1] = 1.0
        
    L[0,0] = 1.0
    
    return L
    