import optuna
import joblib
import numpy as np
import torch
import os
import argparse
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, RandomSampler
import torch.optim as optim
import json

from utils.utils_dataset import *
from utils.utils_unet import *
from utils.utils_training import *
from utils.utils_metrics import *
from utils.utils_file_handler import *


def get_model_params(trial):
    ''' get model parameters '''
    # define hyperparameter ranges
    model_params = {}
    model_params['init_channels'] = trial.user_attrs['init_channels']
    model_params['filters'] = trial.user_attrs['filters'] #trial.suggest_discrete_uniform("filters", 32 ,64, 16) 
    model_params['output_channels'] = trial.user_attrs['output_channels']
    model_params['model_depth'] = trial.params['model_depth']
    model_params['pool_mode'] = trial.params['pool_mode']
    model_params['up_mode'] =  trial.params['up_mode']
    model_params['connection'] = trial.params['connection']
    model_params['same_padding'] = trial.user_attrs['same_padding']
    model_params['use_batchnorm'] = trial.params['use_batchnorm']
    model_params['use_dropblock'] = trial.params['use_dropblock']
    model_params['conv_layers_per_block'] = trial.user_attrs['conv_layers_per_block']
    model_params['activation_fcn'] = trial.params['activation_fcn']
    
    
    # conditional hyperparameter ranges
    if model_params['use_dropblock']:
        model_params['block_size'] = trial.params['block_size']
        model_params['max_drop_prob'] = trial.params['max_drop_prob']
        model_params['dropblock_ls_steps'] = trial.params['dropblock_ls_steps']
    if model_params['activation_fcn'] == 'leaky_relu':
        model_params['neg_slope'] = trial.params['neg_slope']
    
    return model_params

def get_general_params(trial):
    
    # define other training params
    general_params = {}
    general_params['num_epochs'] = trial.user_attrs['num_epochs'] # from previous tests, all we'll need
    general_params['train_batch_size'] = trial.params['train_batch_size'] # will we overflow on memory
    general_params['val_batch_size'] = trial.user_attrs['val_batch_size']
    general_params['test_batch_size'] = trial.user_attrs['test_batch_size']
    general_params['initialization'] = trial.params['initialization']
    
    return general_params

def get_optimizer(trial, network_params):
    
    optimizer = trial.params['optimizer']
    lr = trial.params['lr']
    if optimizer == 'sgd':
        momentum = trial.user_attrs["momentum"]
        return optim.SGD(network_params, lr = lr, momentum = momentum)
    elif optimizer == 'rmsprop':
        momentum = trial.user_attrs["momentum"]
        return optim.RMSprop(network_params, lr = lr, momentum = momentum)
    elif optimizer == 'adam':
        return optim.Adam(network_params, lr)



def training(trial, train = True):
    ''' our objective function for training '''


    model_params = get_model_params(trial)
    general_params = get_general_params(trial)
    
    
    # save path
    save_path = 'results'
    
      # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    # transformations
    train_transform = transforms.Compose([AutoPad(original_size = (584, 565), model_depth = model_params['model_depth'], fill = 0, padding_mode = "constant"),
                                     ToTensor()])

    test_transform = transforms.Compose([AutoPad(original_size = (584, 565), model_depth = model_params['model_depth'], fill = 0, padding_mode = "constant"),
                                     ToTensor()])
    
    train_root = 'final_data/augmented_training/'
    val_root = 'final_data/gray_validation/'
    test_root = 'final_data/gray_test/'
    # retrieve dataset
    train_dataset = CustomDataset(image_root=train_root + 'images',
                                  target_root=train_root  + 'targets',
                                  mask_root=train_root + 'masks',
                                  transform=train_transform)
    val_dataset = CustomDataset(image_root=val_root + 'images',
                                  target_root=val_root + 'targets',
                                  mask_root=val_root + 'masks',
                                  transform=train_transform)
    test_dataset = CustomDataset(image_root=test_root + 'images',
                                 transform = test_transform)

    # batch sizes
    train_batch_size = general_params['train_batch_size']
    val_batch_size = general_params['val_batch_size'] 
    test_batch_size = general_params['test_batch_size']
    

    # load into dataloaders
    train_loader = DataLoader(train_dataset, batch_size = train_batch_size,shuffle=True, drop_last=False) 
    val_loader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False )
    
    # check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    # set up unet
    unet = UNet(**model_params)
    unet.to(device)
    #print(torch.cuda.memory_summary(device))

    # optimizer / optimizer parameters
    params = unet.parameters()
    optimizer = get_optimizer(trial, params)
    
    # loss function
    loss_fn = nn.BCELoss()

    
    # test initial weights
    if general_params['initialization'] == 'unet':
        unet.apply(unet_initialization)
    elif general_params['initialization'] == 'kaiming_norm':
        unet.apply(init_weight_k_norm)
    elif general_params['initialization'] == 'kaiming_uni':
        unet.apply(init_weight_k_uni)
    elif general_params['initialization'] == 'random':
        pass # do nothing
    
    values = {'train_loss': [], 'val_loss': []}
    if train:
      # Actual training loop
      num_epochs = general_params['num_epochs']
      for epoch in range(1, num_epochs + 1):
          #train
        
          train_loss = train_epoch(epoch=epoch,
                                  network=unet,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  dataloader=train_loader,
                                  device=device,
                                  use_mask=True,
                                  debug = False,
                                  verbose=True)
          #validate
          val_loss = val_epoch(epoch=epoch,
                            network=unet,
                            loss_fn=loss_fn,
                            dataloader=val_loader,
                            device=device,
                            use_mask=True,
                            verbose=True)  

          # save model at best val loss...
          if len(values['val_loss']) >= 1 and val_loss < values['val_loss'][-1]:
            torch.save(unet.state_dict(), os.path.join(save_path, "model.pth"))
          elif len(values['val_loss']) < 1:
            torch.save(unet.state_dict(), os.path.join(save_path, "model.pth"))
              

          #report metrics
          values['train_loss'].append( train_loss)
          values['val_loss'].append(val_loss)

    final_test_metrics(network=unet,
                   val_dataloader=val_loader,
                   test_dataloader=test_loader,
                   train_losses=values['train_loss'],
                   val_losses=values['val_loss'],
                   device=device,
                   use_mask=True,
                   num_test_samples=20,
                   save_path=save_path)
  
  

    return 
  
  

  
if __name__ == '__main__':
  # load study
  study = joblib.load('results/study.pkl')

  if os.path.exists('results/model.pth'): # if the model is already saved, then no need to retrain
      train = False
  else:
      train = True
      
  # save parameters
  all_params = {}
  all_params['optimized'] = study.best_trial.params
  all_params['constant'] = study.best_trial.user_attrs
  with open('results/model_params', 'w') as outfile:
    json.dump(all_params, outfile)
  #train model w/ best trial
  training(study.best_trial, train = train)
  
  

