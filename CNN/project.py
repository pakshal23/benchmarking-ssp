import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import json
import matplotlib.pyplot as plt
import numpy as np
import io


class Project():

    def __init__(self, params):

        self.params = params

        
    def init_log(self):
        """ Create log directory as: self.params.log_dir
            Initialize the self.writer as a SummaryWriter
        """
        
        self.log_folder = self.params.log_dir + '/' + self.params.network.net + '_F_' + str(self.params.network.filter_size) + '_C_' + str(self.params.network.num_channels) + '_L_' + str(self.params.network.num_layers) + '_batchsize_' + str(self.params.dataloader.batch_size) + '_lr_' + str(self.params.optimizer.lr) + '_num_samples_' + str(self.params.dataset.num_train) + '_wd_' + str(self.params.optimizer.weight_decay)

        if not os.path.isdir(self.log_folder):
            os.makedirs(self.log_folder)
            os.makedirs(self.log_folder + '/models')
            
            
        # save the configurations in the log directory
        with open(self.log_folder + '/configuration.json', 'w') as fp:
            json.dump(self.params, fp, indent=4)

        if self.params.tensorboard:
            # init summary writer
            self.writer = SummaryWriter(self.log_folder + '/tensorboard')


    def init_devices(self):
        """ """
        if (self.params.device == 'cuda:0') or (self.params.device == 'cuda:1') or (self.params.device == 'cuda:2') or (self.params.device == 'cuda:3'):
            if torch.cuda.is_available():
                self.device = self.params.device
                print('\nUsing GPU.')
            else:
                self.device = 'cpu'
                print('\nCUDA not available. Using CPU.')
        else:
            self.device = 'cpu'
            print('\nUsing CPU.')

            
            
    def log_step_training(self, epoch_loss, epoch):
        
        print('[{:d}]'.format(epoch), end=' ')
        print('{}: {:7.3f}'.format("Epoch loss", epoch_loss))
        
        if self.params['tensorboard']:
            self.writer.add_scalars("Training/", {"Training epoch loss" : epoch_loss}, epoch)

    
    
    def log_step_validation(self, valid_loss, epoch):
        
        print('[{:d}]'.format(epoch), end=' ')
        print('{}: {:7.3f}'.format("Validation loss", valid_loss))
        
        if self.params['tensorboard']:
            self.writer.add_scalars("Validation/", {"Validation loss" : valid_loss}, epoch)



    def log_step_test(self, test_loss, epoch):
        
        print('[{:d}]'.format(epoch), end=' ')
        print('{}: {:7.3f}'.format("Test loss", test_loss))
        
        if self.params['tensorboard']:
            self.writer.add_scalars("Test/", {"Test loss" : test_loss}, epoch)
            
    
            
    def log_signals(self, init_reconstructions, reconstructed_signals, ground_truth, epoch):
        
        # Create a list of matplotlib figures
        figures_list = []
        num_figures = reconstructed_signals.shape[0]
        K = reconstructed_signals.shape[1]
        xrange = np.arange(K)
        for kk in range(num_figures):
            init_sig = init_reconstructions[kk,:]
            recon_sig = reconstructed_signals[kk,:]
            gt = ground_truth[kk,:]
            fig = plt.figure()
            ax = plt.gca()
            ax.plot(xrange, gt.numpy(), color='green')
            ax.plot(xrange, init_sig.numpy(), color='red')
            ax.plot(xrange, recon_sig.numpy(), color='blue')
            figures_list.append(fig)
            #plt.savefig(self.params.log_dir + '/fig1')
            
        if self.params['tensorboard']: 
            self.writer.add_figure('Validation set signals/', figures_list, global_step=epoch)
                        
            
            
    def save_model(self, epoch):
        
        state = {
            'model_state'        : self.net.state_dict(),
            'optimizer_state'    : self.optimizer.state_dict(),
            'num_epochs_finished'    : epoch
        }

        torch.save(state, self.log_folder + '/models/model_' + str(epoch).zfill(6) + '.pt')