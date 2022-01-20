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
import shutil

from utils.utils_dataset import *
from utils.utils_unet import *
from Unet_research.unet_code.utils.utils_training_old import *
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

def get_optimizer_ini(ini_dict, network_params):
    optimizer = ini_dict['optimizer']
    lr = ini_dict['lr']
    if optimizer == 'sgd':
        momentum = ini_dict["momentum"]
        return optim.SGD(network_params, lr = lr, momentum = momentum)
    elif optimizer == 'rmsprop':
        momentum = ini_dict["momentum"]
        return optim.RMSprop(network_params, lr = lr, momentum = momentum)
    elif optimizer == 'adam':
        return optim.Adam(network_params, lr)



def training(trial = None,ini_file = None, train = True,save_path = 'results', im_size = (584, 565)):
    ''' our objective function for training '''

    if ini_file:
        model_params = read_config( ini_file, 'model_params')
        general_params = read_config( ini_file, 'general_params')
    elif trial:
        model_params = get_model_params(trial)
        general_params = get_general_params(trial)
    
      # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    # transformations
    train_transform = transforms.Compose([AutoPad(original_size = im_size, model_depth = model_params['model_depth'], fill = 0, padding_mode = "constant"),
                                     ToTensor()])

    test_transform = transforms.Compose([AutoPad(original_size = im_size, model_depth = model_params['model_depth'], fill = 0, padding_mode = "constant"),
                                     ToTensor()])
    
    train_root = f'final_data_{im_size[0]}_{im_size[1]}/augmented_training/'
    val_root = f'final_data_{im_size[0]}_{im_size[1]}/gray_validation/'
    test_root = f'final_data_{im_size[0]}_{im_size[1]}/gray_test/'
    # retrieve dataset
    train_dataset = CustomDataset(image_root=train_root + 'images',
                                  target_root=train_root  + 'targets',
                                  mask_root=train_root + 'masks',
                                  transform=train_transform)
    val_dataset = CustomDataset(image_root=val_root + 'images',
                                  target_root=val_root + 'targets',
                                  mask_root=val_root + 'masks',
                                  transform=test_transform)
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
    
    if ini_file:
        optimizer = get_optimizer_ini(read_config(ini_file, 'optimizer_params'), params)
    elif trial:
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
          val_loss = val_epoch_dropblock(epoch=epoch, # use the dropblock version
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

    else:
        # load parameters from the save_file to pretrained model
        # already done in final_test_metrics
        pass
    
    # load best saveed model
    unet.load_state_dict(torch.load(save_path + '/model.pth',  map_location=torch.device(device)))

        
    final_test_metrics(network=unet,
                   val_dataloader=val_loader,
                   test_dataloader=test_loader,
                   train_losses=values['train_loss'],
                   val_losses=values['val_loss'],
                   device=device,
                   use_mask=True,
                   num_test_samples=20,
                   save_path=save_path,
                   im_size=im_size)
  
  

    return 
  
  

  
if __name__ == '__main__':
    
    # if args exist, then don't use study
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='inifile', help = 'Ini file containing parameters to use' )
    parser.add_argument('-s', dest ='saveloc', default = 'results', help = 'save location of files')
    parser.add_argument('-t', dest='train', action = 'store_true')
    parser.add_argument( '-width', dest = 'width', type = int, default = 565, help = 'Width of final images')
    parser.add_argument('-height', dest = 'height', type = int, default = 584, help = 'Heigh of final images')
    
    args = parser.parse_args()
    inifile = args.inifile
    train = args.train
    save_path = args.saveloc
    im_width = args.width
    im_height = args.height
    
    if inifile:
        # params will be known from inifile, placed in constant
        all_params = {}
        model_params = read_config( inifile, 'model_params')
        general_params = read_config( inifile, 'general_params')
        
        all_params['constant'] = {**model_params, **general_params} #combine all data
        with open(save_path + '/model_params', 'w') as outfile:
            json.dump(all_params, outfile)
        shutil.copy(inifile, save_path + '/model_params.ini')
        
        training(ini_file = inifile, train = train, save_path = save_path, im_size=(im_height, im_width))
    else:
        # load study
        study = joblib.load(save_path + '/study.pkl')
        
        # save parameters
        all_params = {}
        all_params['optimized'] = study.best_trial.params
        all_params['constant'] = study.best_trial.user_attrs
        with open(save_path + '/model_params', 'w') as outfile:
            json.dump(all_params, outfile)
        #train model w/ best trial
        training(trial = study.best_trial, train = train, save_path = save_path, im_size=(im_height, im_width))
  
  

