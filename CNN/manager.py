# References :
# https://github.com/kuangliu/pytorch-cifar
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from project import Project
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import init_dataset
from networks import ResCNN, ResUNet
from utils import get_data_loader


#### MANAGER

class Manager(Project):
    """ """

    def __init__(self, params):
        """ """

        super().__init__(params)

        self.init_devices()
        self.init_log()
    
        # Display the parameters
        print('\nconfiguration : ', self.params, sep='\n')
        
        # Construct the dataset object
        self.dataset = init_dataset(self.params.dataset)        
        
        # Build the network
        self.net = self.build_model(self.params.network, self.device)
        
        # Set up the optimization for training
        if (self.params.mode == 'train'):
            self.set_optimization(self.params.optimizer)
        
        # Define the loss functions
        self.criterion = nn.MSELoss(reduction='sum')
        self.criterion.to(self.device)
        
        return

    
    
    @staticmethod
    def build_model(params, device='cpu'):
        """ """
        print('\n==> Building model..')
        custom_models_dict = {  
                                'rescnn' : ResCNN,
                                'resunet': ResUNet
                             }

        assert params.net in custom_models_dict.keys()
        net = custom_models_dict[params.net](params, device)
        net = net.to(device)
 
        print('\n[Network] Total number of parameters : {}'.format(net.get_num_params()))

        return net
        
        
        
    def set_optimization(self, params):
        """ """
        self.optimizer = None
        
        optim_name = params.name 
        lr, weight_decay = params.lr, params.weight_decay
        
        conv_params_iter = self.net.parameters_conv()
        bn_params_iter = self.net.parameters_bn()
        
        # Set the optimizer
        if optim_name == 'adam':
            self.optimizer = optim.Adam([ {'params' : conv_params_iter}, 
                                         {'params' : bn_params_iter, 'weight_decay' : 0.0}], lr = lr, weight_decay = weight_decay)
    
        elif optim_name == 'sgd':
            self.optimizer = optim.SGD([ {'params' : conv_params_iter},                                            
                                        {'params' : bn_params_iter, 'weight_decay' : 0.0}], lr = lr, weight_decay = weight_decay, momentum = 0.9, nesterov=True)
            
        # Set the scheduler
        self.scheduler = None
        if (params.lr_scheduler):
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, params.milestones, gamma=params.lr_gamma)
        
        


##################################################################################################
#### TRAIN

    def train(self):
        """ """
        self.net.train()

        print('\n==> Preparing data..')
        self.trainset, self.forward_model = self.dataset.get_dataset(dataset_type='train')
        self.trainloader = get_data_loader(self.trainset, self.params.dataloader)
        self.validset, temp_forward_model = self.dataset.get_dataset(dataset_type='valid')
        self.validloader = get_data_loader(self.validset, self.params.dataloader)
        
        # Set the forward model in the network (for initial reconstruction)
        self.net.set_forward_model(self.forward_model)
            
        print('\n\nStarting training...')
        for epoch in range(self.params.optimizer.num_epochs):
            epoch_loss = 0.0

            for batch_idx, (measurements, signals) in enumerate(self.trainloader):
            
                measurements, signals = measurements.to(self.device), signals.to(self.device)
                reconstructed_signals = self.net(measurements)
                batch_loss = (1.0/self.params.dataloader.batch_size)*self.criterion(reconstructed_signals, signals)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss = epoch_loss + self.params.dataloader.batch_size*batch_loss.item()
            
            mean_epoch_loss = epoch_loss/self.params.dataset.num_train
            self.log_step_training(mean_epoch_loss, epoch)
            if (epoch % self.params.validation_step == 0): 
                self.validation_step(epoch)
            self.net.train()
        
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            
        print('\nFinished training.')
        
        return 



    def validation_step(self, epoch):
        """ """
        self.net.eval()
        valid_loss = 0.0
        with torch.no_grad():

            for batch_idx, (measurements, signals) in enumerate(self.validloader):

                measurements, signals = measurements.to(self.device), signals.to(self.device)
                initial_reconstructed_signals = self.net.get_initial_reconstruction(measurements)
                reconstructed_signals = self.net(measurements)

                batch_valid_loss = self.criterion(reconstructed_signals, signals)
                valid_loss += batch_valid_loss.item()
        
            mean_valid_loss = valid_loss/self.params.dataset.num_valid

            self.log_step_validation(mean_valid_loss, epoch)
            # Log the last batch of signals
            self.log_signals(initial_reconstructed_signals[0:10,:].cpu(), reconstructed_signals[0:10,:].cpu(), signals[0:10,:].cpu(), epoch)
            self.save_model(epoch)

        return

    
    
    def eval_dataset_mse(self, mode="train"):
        """ """
        self.net.eval()
        self.dataset_eval, forward_model = self.dataset.get_dataset(dataset_type=mode)
        self.dataloader_eval = get_data_loader(self.dataset_eval, self.params.dataloader)
        self.net.set_forward_model(forward_model)
        
        if (mode == "train"):
            num_signals = self.params.dataset.num_train
        elif (mode == "valid"):
            num_signals = self.params.dataset.num_valid
        elif (mode == "test"):
            num_signals = self.params.dataset.num_test
            
        total_err = 0.0
        with torch.no_grad():

            for batch_idx, (measurements, signals) in enumerate(self.dataloader_eval):

                measurements, signals = measurements.to(self.device), signals.to(self.device)
                reconstructed_signals = self.net(measurements)

                batch_err = self.criterion(reconstructed_signals, signals)
                total_err += batch_err.item()
                
            mse = total_err/num_signals
            
        return mse