from numpy import save
import torch
import os
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, RandomSampler
import torch.optim as optim
import logging

from utils.utils_dataset import *
from utils.utils_unet import *
from utils.utils_training import *
from utils.utils_metrics import *


# set up save file
if not os.path.exists("metrics/test"):
  os.mkdir("metrics/test")
  
  
# set up a logger
logging.basicConfig(filename="metrics/test/console.log", 
					format= '%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s', ) 
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


root = '.'

# Retrieve data and include transformation operations
logger.info('Retrieving Data')

train_path_images = root + '/datasets/training/images'
train_path_target = root + '/datasets/training/1st_manual'
train_path_mask = root + '/datasets/training/mask'
test_path = root + '/datasets/test/images'

# transformations

# alternative using Random Operations function
'''
train_transform = transforms.Compose([RandomOperations([RandomHorizontalFlip(p = 1),
                                                        RandomVerticalFlip(p = 1)],
                                                       weights = [.5, .5]),
                                      RandomRotate(degrees = 180),
                                      ToTensor()])
'''
train_transform = transforms.Compose([AutoPad(original_size = (584, 565), model_depth = 4, fill = 0, padding_mode = "constant"),
                                     RandomHorizontalFlip(p = .5),
                                     RandomVerticalFlip(p = .5),
                                     RandomRotate(degrees = 180),
                                     ToTensor()])

test_transform = transforms.Compose([AutoPad(original_size = (584, 565), model_depth = 4, fill = 0, padding_mode = "constant"),
                                     ToTensor()])

# retrieve dataset
train_dataset = CustomDataset(image_root=train_path_images,
                              target_root=train_path_target,
                              mask_root=train_path_mask,
                              transform=train_transform)
test_dataset = CustomDataset(image_root=test_path,
                             transform = test_transform)


# Put data into dataloaders
logger.info('Loading Data')

# batch sizes
train_batches_per_epoch = 14
train_batch_size = 1
train_samples = train_batches_per_epoch * train_batch_size

val_batch_size = 1 # validate in one batch
test_batch_size = 1 # test one by one

# split into train and val
train_size = int(len(train_dataset) * .7)
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])


# load into dataloaders
#rand_sampler_train = RandomSampler(train_data, num_samples=train_samples, replacement=True) # sample with replacement
train_loader = DataLoader(train_data, batch_size = train_batch_size,)# sampler = rand_sampler_train)

val_loader = DataLoader(val_data, batch_size = val_batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False )


# set up UNET and Optimizers

logger.info('Setting up UNet')

# check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
logger.info(f'Device:  {device}')
'''
unet = UNetold(init_channels = 3,
            filters = 64,
            output_channels = 2,
            pool_mode = 'max',
            up_mode = 'upconv', 
            connection = 'cat',
            same_padding = True,
            use_batchnorm = True,
            conv_layers_per_block = 2
            )
'''
unet = UNet(init_channels = 3,
            filters = 64,
            output_channels = 2,
            model_depth = 4,
            pool_mode = 'max',
            up_mode = 'upconv', 
            connection = 'cat',
            same_padding = True,
            use_batchnorm = True,
            use_dropblock = False,
            block_size = 7,
            drop_prob = .1,
            conv_layers_per_block = 2,
            activation_fcn = 'leaky_relu',
            neg_slope = 0.01,
            )


unet.to(device)

# !!!!! SET UP INITIAL WEIGHTS according to article
unet.apply(unet_initialization)

# optimizer parameters
params = unet.parameters()
lr = .001
momentum = .99 # according to article
optimizer = optim.SGD(params, lr = lr, momentum = momentum )

# loss function
loss_fn = nn.BCELoss()

# LR scheduler (for later )

'''
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min',
                                                 factor=.1,
                                                 threshold=.005,
                                                 patience=5,
                                                 verbose=True)
'''

# training cycle
logger.info('Start Training Cycle')

# validate once on random initialization
val_loss = val_epoch(epoch=0,
                     network=unet,
                     loss_fn=loss_fn,
                     dataloader=val_loader,
                     device=device,
                     use_mask=True,
                     logger=logger)
logger.info(f'No Training Validation Loss - {val_loss}')


num_epochs = 1
values = {'train_loss': [], 'val_loss': []}
for epoch in range(1, num_epochs + 1):
  #train

  train_loss = train_epoch(epoch=epoch,
                           network=unet,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           dataloader=train_loader,
                           device=device,
                           use_mask=True,
                           logger=logger)
  #validate
  val_loss = val_epoch(epoch=epoch,
                     network=unet,
                     loss_fn=loss_fn,
                     dataloader=val_loader,
                     device=device,
                     use_mask=True,
                     logger=logger)  
  #report metrics
  logger.info(f'Train Loss: {train_loss}')
  logger.info(f'Val Loss: {val_loss}')
  values['train_loss'].append(train_loss)
  values['val_loss'].append(val_loss)
  
final_test_metrics(network=unet,
                   val_dataloader=val_loader,
                   test_dataloader=test_loader,
                   train_losses=values['train_loss'],
                   val_losses=values['val_loss'],
                   device=device,
                   use_mask = True,
                   num_test_samples = 5, # conserve space
                   save_model = False,
                   save_path = "./metrics/test")
  
  
  # change lr
  #scheduler.step(val_loss)