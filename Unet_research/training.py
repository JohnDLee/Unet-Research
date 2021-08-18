from numpy import save
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from utils.utils_dataset import *
from utils.utils_unet import *
from utils.utils_training import *
from utils.utils_logger import logger




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
train_transform = transforms.Compose([RandomHorizontalFlip(p = .5),
                                     RandomVerticalFlip(p = .5),
                                     RandomRotate(degrees = 180),
                                     ToTensor()])

test_transform = ToTensor()

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
train_batch_size = 1
val_batch_size = 1
test_batch_size = 1

# split into train and val
train_size = int(len(train_dataset) * .7)
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

# load into dataloaders
train_loader = DataLoader(train_data, batch_size = train_batch_size, shuffle = False)
val_loader = DataLoader(val_data, batch_size = val_batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True )

# set up UNET and Optimizers

logger.info('Setting up UNet')

# check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
logger.info('Device:', device)

unet = UNet(init_channels = 3,
            filters = 64,
            output_channels = 2,
            pool_mode = 'max',
            up_mode = 'upconv', 
            connection = 'cat',
            same_padding = True,
            use_batchnorm = True,
            conv_layers_per_block = 2
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
                                                 patience=5,
                                                 verbose=True)
'''

# training cycle
logger.info('Training')

# validate once on random initialization
val_loss = val_epoch(epoch=0,
                     network=unet,
                     loss_fn=loss_fn,
                     dataloader=val_loader,
                     device=device,
                     use_mask=True)
logger.info(f'No Training Validation Loss - {val_loss}')


num_epochs = 30
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
                           save_location='./models/')
  #validate
  val_loss = val_epoch(epoch=0,
                     network=unet,
                     loss_fn=loss_fn,
                     dataloader=val_loader,
                     device=device,
                     use_mask=True)  
  #report metrics
  logger.info(f'Train Loss: {train_loss}\nVal Loss: {val_loss}')
  values['train_loss'].append(train_loss)
  values['val_loss'].append(val_loss)
  
  test_epoch(epoch=epoch,
             network=unet,
             dataloader=test_loader,
             device = device,
             num_cols=5,
             save=True,
             save_location='./models/'
             )
  
  
  # change lr
  #scheduler.step(val_loss)