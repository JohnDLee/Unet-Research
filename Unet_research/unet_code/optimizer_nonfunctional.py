
import numpy as np
import joblib
import optuna
import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from Unet_research.unet_code.utils.utils_training_old import *
from utils.utils_dataset import *
from utils.utils_unet import *
from utils.utils_metrics import *
from utils.utils_file_handler import *


def get_model_params(trial):
    ''' get model parameters '''
    # define hyperparameter ranges
    model_params = {}
    model_params['init_channels'] = 3 # change this to 1
    trial.set_user_attr("init_channels", model_params['init_channels'])
    model_params['filters'] = 64 #trial.suggest_discrete_uniform("filters", 32 ,64, 16) 
    trial.set_user_attr("filters", model_params['filters'])
    model_params['output_channels'] = 2
    trial.set_user_attr("output_channels", model_params['output_channels'])
    model_params['model_depth'] = trial.suggest_int('model_depth', low = 2, high = 6, step = 1)
    model_params['pool_mode'] = trial.suggest_categorical('pool_mode', ["max", "conv"]) # use Max Pool
    #model_params['pool_mode'] = trial.suggest_categorical('pool_mode', ["max", "conv"])
    model_params['up_mode'] = trial.suggest_categorical('up_mode', ["upconv", "upsample"])
    model_params['connection'] = trial.suggest_categorical('connection', ["cat", "add"])
    model_params['same_padding'] = True
    trial.set_user_attr("same_padding", model_params['same_padding'])
    model_params['use_batchnorm'] = trial.suggest_categorical('use_batchnorm', [True]) # Must use BatchNorm
    model_params['use_dropblock'] = trial.suggest_categorical('use_dropblock', [True]) # Must use Dropblock
    #model_params['use_dropblock'] = trial.suggest_categorical('use_dropblock', [True, False])
    #model_params['use_batchnorm'] = trial.suggest_categorical('use_batchnorm', [True, False])
    model_params['conv_layers_per_block'] = 2
    trial.set_user_attr("conv_layers_per_block", model_params['conv_layers_per_block'])
    model_params['activation_fcn'] = trial.suggest_categorical('activation_fcn',['relu', 'leaky_relu'])
    
    
    # conditional hyperparameter ranges
    if model_params['use_dropblock']:
        model_params['block_size'] = trial.suggest_int('block_size', low = 2, high = 11, step = 1)
        model_params['max_drop_prob'] = trial.suggest_float('max_drop_prob', low = .1, high = .25,)
        model_params['dropblock_ls_steps'] = trial.suggest_int('dropblock_ls_steps', low = 500, high = 3000)
    if model_params['activation_fcn'] == 'leaky_relu':
        model_params['neg_slope'] = trial.suggest_float('neg_slope', 1e-4, 1e-1, log = True)
    
    return model_params

def get_general_params(trial):
    
    # define other training params
    general_params = {}
    general_params['num_epochs'] = 50 # from previous tests, all we'll need
    trial.set_user_attr("num_epochs", general_params['num_epochs'])
    general_params['train_batch_size'] = trial.suggest_int('train_batch_size', 1, 4,) 
    general_params['val_batch_size'] = 1
    trial.set_user_attr("val_batch_size", general_params['val_batch_size'])
    general_params['test_batch_size'] = 1
    trial.set_user_attr("test_batch_size", general_params['test_batch_size'])
    general_params['initialization'] = trial.suggest_categorical('initialization', ['unet', 'kaiming_norm', 'kaiming_uni', 'random'])
    
    return general_params

def get_optimizer(trial, network_params):
    
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'rmsprop'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log = True)
    momentum = .99 # default
    if optimizer == 'sgd':
        trial.set_user_attr("momentum", momentum)
        return optim.SGD(network_params, lr = lr, momentum = momentum)
    elif optimizer == 'rmsprop':
        trial.set_user_attr("momentum", momentum)
        return optim.RMSprop(network_params, lr = lr, momentum = momentum)
    elif optimizer == 'adam':
        return optim.Adam(network_params, lr)

    
def param_optimizer(trial):
    ''' our objective function for optimizing '''
    
    # get parameters for our runs
    model_params = get_model_params(trial)
    general_params = get_general_params(trial)
    
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    # cuDNN autonuer to select best kernel
    torch.backends.cudnn.benchmark = True


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
    
    # split dataset to decrease size for faster optimizations
    halved_dataset, _ = random_split(train_dataset, [int(.5 * len(train_dataset)), int(len(train_dataset) - int(.5 * len(train_dataset)))], generator=torch.Generator().manual_seed(0))
    
    # load into dataloaders
    train_loader = DataLoader(halved_dataset, batch_size = train_batch_size,shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False )


    # check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # set up unet
    unet = UNet(**model_params)
    unet.to(device)

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
                               use_mask=True,)
        #validate
        val_loss = val_epoch(epoch=epoch,
                         network=unet,
                         loss_fn=loss_fn,
                         dataloader=val_loader,
                         device=device,
                         use_mask=True,)  

        
        trial.report(val_loss, epoch) # report the val_loss to see if we should prune

        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss # BCE Loss is our metric


class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()
            
# if still getting issue


if __name__ == '__main__':
    # parse arguments
    parser = setup_argparser()
    num_trials = parser.parse_args().num_trials[0]
    print(num_trials)
    
    # create a new study
    study = optuna.create_study(sampler = optuna.samplers.TPESampler(seed = 0), pruner = optuna.pruners.HyperbandPruner(), study_name = 'unet',storage='sqlite:///results/unet.db')
    
    pruned_callback = StopWhenTrialKeepBeingPrunedCallback(10)
    # if we run out of mem, ignore
    # if we near 7 days, exit early
    study.optimize(param_optimizer, n_trials=num_trials, catch = (RuntimeError,), callbacks = [pruned_callback], gc_after_trial = True, timeout = 500000) 
    
    # save optimization metrics too!
    save_optimization_metrics(study, save_path = 'results')
    
    # save they study itself
    save_path = 'results'
    joblib.dump(study, os.path.join(save_path, 'study.pkl'))
    