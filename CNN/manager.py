from project import Project
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import init_dataset
from networks import ResCNN, ResCNN_no_bn
from utils import get_data_loader
import time


#### MANAGER

class Manager(Project):
    """ """

    def __init__(self, params):
        """ """

        super().__init__(params)

        self.init_devices()

        if (self.params.mode == 'train'):
            self.init_log()
    
        # Display the parameters
        print('\nconfiguration : ', self.params, sep='\n')
        
        # Construct the dataset object
        self.dataset = init_dataset(self.params.dataset)        
        
        # Build the network
        self.net = self.build_model(self.params.network, self.device)

        if (self.params.mode == 'test'):
            self.net.eval()
            model_dir = self.params.log_dir + '/' + self.params.network.net + '_F_' + str(self.params.network.filter_size) + '_C_' + str(self.params.network.num_channels) + '_L_' + str(self.params.network.num_layers) + '_batchsize_' + str(self.params.dataloader.batch_size) + '_lr_' + str(self.params.optimizer.lr) + '_num_samples_' + str(self.params.dataset.num_train) + '_wd_' + str(self.params.optimizer.weight_decay)
            #model_dir = self.params.log_dir + '/' + self.params.network.net + '_F_' + str(self.params.network.filter_size) + '_C_' + str(self.params.network.num_channels) + '_L_' + str(self.params.network.num_layers) + '_batchsize_' + str(self.params.dataloader.batch_size) + '_lr_' + str(self.params.optimizer.lr) + '_num_samples_' + str(self.params.dataset.num_train)
            model_file = model_dir + '/models/model_' + str(self.params.model_num).zfill(6) + '.pt'
            loaded_dict = torch.load(model_file, map_location='cpu')
            self.net.load_state_dict(loaded_dict['model_state'])
        
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
                                'rescnn_no_bn' : ResCNN_no_bn
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
        
        # Set the optimizer
        if optim_name == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr = lr, weight_decay = weight_decay)
    
        elif optim_name == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr = lr, weight_decay = weight_decay, momentum = 0.9, nesterov=True)
            
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

                #print(signals.shape[0])
            
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
                self.scheduler.step()
            
        print('\nFinished training.')
        
        return 



    def validation_step(self, epoch):
        """ """
        self.net.eval()
        valid_loss = 0.0
        with torch.no_grad():

            #print('Validation')

            for batch_idx, (measurements, signals) in enumerate(self.validloader):

                #print(signals.shape[0])

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
        self.params.dataloader.batch_size = 1
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
        #total_time = 0.0
        with torch.no_grad():

            for batch_idx, (measurements, signals) in enumerate(self.dataloader_eval):

                print(batch_idx)

                measurements, signals = measurements.to(self.device), signals.to(self.device)
                #t_s = time.time()
                #reconstructed_signals, t_signal = self.net(measurements)
                #total_time = total_time + t_signal

                reconstructed_signals = self.net(measurements)

                batch_err = self.criterion(reconstructed_signals, signals)
                total_err += batch_err.item()
                
            mse = total_err/num_signals
            #mean_time = total_time/num_signals
            
        #return mse, mean_time
        return mse