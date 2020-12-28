#!/usr/bin/env python3

import argparse
from utils import get_config_from_json
from manager import Manager


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Neural Network Reconstruction for Sparse Stochastic Processes')
    parser.add_argument('--config_file', default="./configs/configs_1.json", help="Specify the configuration file", metavar="FILE")
    args = parser.parse_args()
    params = get_config_from_json(args.config_file) # transform to dictionary
    
    if (params.mode == 'train'):
        manager_obj = Manager(params)
        # Train the model
        manager_obj.train()
        # Get the training and validation set MSE for the trained network
        training_set_mse = manager_obj.eval_dataset_mse(mode='train')
        print('Training error: ' + str(training_set_mse))
        validation_set_mse = manager_obj.eval_dataset_mse(mode='valid')
        print('Validation error: ' + str(validation_set_mse))
        
    elif (params.mode == 'test'):
        # In the "test" mode , please provide a file with the trained model as "model_dir"
        manager_obj = Manager(params)
        # Evaluate the provided trained model for the test dataset
        test_set_mse = manager_obj.eval_dataset_mse(mode='test')
        print('Testing error: ' + str(test_set_mse))