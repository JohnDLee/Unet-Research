import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from os.path import join, exists
from PIL import Image #image reader
import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import sys

sys.path.append(os.path.join(os.getcwd(), 'unet_code'))

from utils.utils_unet import UNet
from utils.utils_modules import DropBlock2D
from utils.utils_training import BaseUNetTraining
from utils.utils_metrics import final_test_metrics
from utils.utils_dataset import UnetDataset


def training(args):

    # seed
    if args.seed is not None:
        seed_everything(args.seed, workers = True)

    # create destination
    dest = args.save_path
    if not exists(dest):
        os.mkdir(dest)
    else:
        for i in range(6):
            dest = dest + str(i)
            if not exists(dest):
                os.mkdir(dest)
                break
        else:
            print("Could not create directory.")
            exit(1)
    
    # get data
    train_root = join(args.data_path, 'train')
    val_root = join(args.data_path, 'val')
    test_root = join(args.data_path, 'test')

    add_images = lambda x: join(x, 'images')
    add_targets = lambda x: join(x, 'targets')
    add_masks = lambda x: join(x, 'masks')

    # datasets
    train_dataset = UnetDataset(image_root=add_images(train_root),
                                target_root=add_targets(train_root),
                                mask_root=add_masks(train_root),
                                mode = {'image': 'L', 'target': 'L', 'mask' : 'L'})
    val_dataset = UnetDataset(image_root=add_images(val_root),
                                target_root=add_targets(val_root),
                                mask_root=add_masks(val_root),
                                mode = {'image': 'L', 'target': 'L', 'mask' : 'L'})
    test_dataset = UnetDataset(image_root=add_images(test_root),
                            mask_root = add_masks(test_root),
                                mode = {'image': 'L', 'target': 'L', 'mask' : 'L'})

    train_batch_size = args.train_batch
    val_batch_size = args.val_batch
    test_batch_size = 1

    # load into dataloaders
    train_loader = DataLoader(train_dataset, batch_size = train_batch_size,shuffle=True, drop_last=False, num_workers=2) 
    val_loader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False, num_workers=2)

    unet = UNet(init_channels=args.input_channels,
                filters=64,
                output_channels=1,
                model_depth=4,
                pool_mode='max',
                up_mode='upconv',
                connection='cat',
                same_padding=True,
                conv_layers_per_block=2,
                checkpointing=True
                )

    unet.set_activation_function(nn.ReLU())
    unet.set_dropblock(DropBlock2D,
                        block_size = args.block_size,
                        drop_prob=args.max_drop_prob,
                        use_scheduler=True,
                        start_drop_prob=0,
                        max_drop_prob=args.max_drop_prob,
                        dropblock_ls_steps=args.dropblock_steps)
    unet.set_normalization(nn.GroupNorm, params = {'num_groups':32, 'num_channels':"fill"})
    unet.create_model()

    # loss function
    loss_fn = nn.BCELoss()

    # optimizer
    optimizer = torch.optim.SGD(params = unet.parameters(), lr = args.lr,momentum =args.momentum )

    model = BaseUNetTraining(unet, loss_fcn=loss_fn, optimizer = optimizer )


    model_info = join(dest, 'model_info')
    os.mkdir(model_info)

    checkpoint_callback = ModelCheckpoint(
                            monitor="val_loss_avg",
                            dirpath=dest,
                            filename="model-{epoch:02d}-{val_loss:.2f}",
                            save_top_k=1,
                            mode="min",
                            )
    trainer = Trainer.from_argparse_args(args, callbacks = [checkpoint_callback], default_root_dir = dest, max_epochs = args.num_epochs)

    # fit model
    trainer.fit(model, train_loader, val_loader)

    statistics = join(dest, 'statistics')
    os.mkdir(statistics)
    final_test_metrics(trainer, model, val_loader, test_loader, save_path = statistics)

    
    




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', dest = 'data_path', required=True, help = 'Datapath containing augmented data. Must contain a train, val, test folder which each have images, (targets), and masks')
    parser.add_argument('-save_path', dest = 'save_path', required = True, help = 'Path to save folder. Should be Nonexistent, but will created a duplicate save_path_X for X = 1-5 if it does.')
    parser.add_argument('-input_channels', dest = 'input_channels', default = 1, type = int, help = 'Number of Input channels. 1 for Grayscale images, 3 for RGB.')
    parser.add_argument('-num_epochs', dest = 'epochs', type = int, default = 50, help = 'Number of Max epochs to run. Defaults to 50')
    parser.add_argument('-train_batch', dest = 'train_batch', type = int, default = 1, help = 'Training batch size. Defaults to 1')
    parser.add_argument('-val_batch', dest = 'val_batch', type = int, default = 1, help = 'Validation batch size. Defaults to 1')
    parser.add_argument('-lr', dest = 'lr', type = float, default = .001, help = 'Optimizer Learning Rate. Defaults to .001')
    parser.add_argument('-momentum', dest = 'momentum', type = float, default = .99, help = 'Momentum the Optimizer will use. Defaults to .99')
    parser.add_argument('-block_size', dest = 'block_size', type = int, default = 7, help = 'Block size of dropblock, which must be odd numbers. A size of 1 is equivalent to dropout. Defaults to 7.')
    parser.add_argument('-max_drop_prob',dest = 'max_drop_prob', type = float, default = .15, help = 'Maximum drop probability of dropblock, must be from 0-1. Defaults to .15')
    parser.add_argument('-dropblock_steps', dest = 'dropblock_steps', type = int, default = 1500, help = 'Number of steps before max drop prob is reached. Defaults to 1500')
    parser.add_argument('-seed', dest = 'seed', type = int, default = None, help = 'Seed for reproducability. Defaults to None' )
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    training(args)