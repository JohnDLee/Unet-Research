import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import os
from os.path import join, exists
from PIL import Image #image reader
import numpy as np
import argparse
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from math import ceil
import sys

sys.path.append(os.path.join(os.getcwd(), 'unet_code'))

from utils.utils_unet import UNet
from utils.utils_modules import DropBlock2D, Dropblock2d_ichan
from utils.utils_training import BaseUNetTraining
from utils.utils_metrics import final_test_metrics
from utils.utils_dataset import UnetDataset
from utils.utils_general import create_dir, square_pad

class RFUNetTraining(BaseUNetTraining):
    """ base training module for UNet, alter predict step for more predictions"""

    def __init__(self, model, loss_fcn, lr, momentum, train_size  = 32):
        super(RFUNetTraining, self).__init__(model, loss_fcn, optimizer = None)
        self.lr = lr
        self.momentum = momentum
        self.train_size = train_size

    def training_step(self, batch, batch_idx):
        im_batch, gt, mask = batch
        im_batch.requires_grad=True


        # pad to square before resizing to 
        im_batch = square_pad(im_batch)
        gt = square_pad(gt)
        mask = square_pad(mask)

        # perform resize on the fly on all data
        im_batch = TF.resize(im_batch, size = (self.train_size, self.train_size))
        gt = TF.resize(gt, size = (self.train_size, self.train_size))
        mask = TF.resize(mask, size = (self.train_size, self.train_size))

        segmentation = self._model(im_batch)

        # mask
        segmentation = segmentation * mask
        gt = gt * mask
        
        loss = self._loss_fcn(segmentation, gt)
        # recalculate loss based on mask
        loss *= (segmentation.numel() / mask.count_nonzero())
        
        # log every 10
        if batch_idx % 10:
            self.log('train_loss', loss, prog_bar=True, logger=True,  on_step = True, on_epoch=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        im_batch, gt, mask = batch

        # pad to square before resizing to 
        im_batch = square_pad(im_batch)
        gt = square_pad(gt)
        mask = square_pad(mask)

        # perform resize on the fly on all data
        im_batch = TF.resize(im_batch, size = (self.train_size, self.train_size))
        gt = TF.resize(gt, size = (self.train_size, self.train_size))
        mask = TF.resize(mask, size = (self.train_size, self.train_size))

        segmentation = self._model(im_batch)

        # mask
        segmentation = segmentation * mask
        gt = gt * mask
        
        loss = self._loss_fcn(segmentation, gt)

        # log
        self.log('val_loss', loss, prog_bar=True, logger=True,  on_step = True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params = self._model.parameters(), lr = self.lr, momentum =self.momentum )
        lr_scheduler_config = {
            'scheduler' :  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                            mode='min',
                            factor=0.1,
                            patience=3,
                            threshold=0.001,
                            threshold_mode='rel', 
                            cooldown=0,
                            min_lr=0,
                            eps=1e-08,
                            verbose=False),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss_epoch',
            'strict': True,
            'name': None,
        }
        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config }

    def predict_step(self, batch, batch_idx):
        im_batch, gt, mask = batch

        # square pad
        im_batch = square_pad(im_batch)
        gt = square_pad(gt)
        mask = square_pad(mask)

        # perform resize on the fly on all data
        im_batch = TF.resize(im_batch, size = (self.train_size, self.train_size))
        gt = TF.resize(gt, size = (self.train_size, self.train_size))
        mask = TF.resize(mask, size = (self.train_size, self.train_size))

        segmentation = self._model(im_batch)

        segmentation = segmentation * mask
        
        return batch_idx, segmentation, im_batch, gt
    

def testing(args):

    # seed
    if args.seed != -1:
        seed_everything(args.seed, workers = True)

    # create new statistics folder if exists
    stats = create_dir(args.save_path)
    if stats is None: # failed
        exit(1)
    

    # get data
    val_root = join(args.data_path, 'val')
    test_root = join(args.data_path, 'test')

    add_images = lambda x: join(x, 'images')
    add_targets = lambda x: join(x, 'targets')
    add_masks = lambda x: join(x, 'masks')

    # datasets
    val_dataset = UnetDataset(image_root=add_images(val_root),
                                target_root=add_targets(val_root),
                                mask_root=add_masks(val_root),
                                mode = {'image': 'L', 'target': 'L', 'mask' : 'L'})
    test_dataset = UnetDataset(image_root=add_images(test_root),
                            mask_root = add_masks(test_root),
                                mode = {'image': 'L', 'target': 'L', 'mask' : 'L'})

    val_batch_size = 1
    test_batch_size = 1

    # load into dataloaders
    val_loader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = False, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False, num_workers=os.cpu_count())

    # set up Unet 
    unet = UNet(init_channels=1,
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
    # Default Dropblock because it should be turned off anyways
    unet.set_dropblock(Dropblock2d_ichan,
                        block_size = args.block_size,
                        drop_prob=args.max_drop_prob,
                        use_scheduler=True,
                        start_drop_prob=0,
                        max_drop_prob=args.max_drop_prob,
                        dropblock_ls_steps=args.dropblock_steps)
    unet.set_normalization(nn.GroupNorm, params = {'num_groups': 32, 'num_channels':"fill"})
    unet.create_model()

    # loss function
    loss_fn = nn.BCELoss()

    # Load Training Lightning Module 
    model = RFUNetTraining.load_from_checkpoint(args.model_path, model=unet, loss_fcn=loss_fn, lr = args.lr, momentum = args.momentum, train_size = args.new_size)

    # call Trainer
    trainer = Trainer.from_argparse_args(args, logger = False)
    
    final_test_metrics(trainer, model, val_loader, test_loader, save_path = stats)
        


def training(args):

    # seed
    if args.seed != -1:
        seed_everything(args.seed, workers = True)

    # create destination
    dest = create_dir(args.save_path)
    if dest is None: # failed
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

        # modify train_dataset to have reduced data.
    if args.train_ratio != 1:
        train_size = ceil(args.train_ratio * len(train_dataset))
        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        del _ # remove extra data.

    train_batch_size = args.train_batch
    val_batch_size = 1
    test_batch_size = 1

    # load into dataloaders
    train_loader = DataLoader(train_dataset, batch_size = train_batch_size,shuffle=True, drop_last=False, num_workers=os.cpu_count()) 
    val_loader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = False, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False, num_workers=os.cpu_count())

    unet = UNet(init_channels=1,
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
    unet.set_dropblock(Dropblock2d_ichan,
                        block_size = args.block_size,
                        drop_prob=args.max_drop_prob,
                        use_scheduler=True,
                        start_drop_prob=0,
                        max_drop_prob=args.max_drop_prob,
                        dropblock_ls_steps=args.dropblock_steps)
    unet.set_normalization(nn.GroupNorm, params = {'num_groups': 32, 'num_channels':"fill"})
    unet.create_model()

    # loss function
    loss_fn = nn.BCELoss()

    # Training Lightning Module
    model = RFUNetTraining(unet, loss_fcn=loss_fn, lr = args.lr, momentum = args.momentum, train_size = args.new_size)


    model_info = join(dest, 'model_info')
    os.mkdir(model_info)

    checkpoint_callback = ModelCheckpoint(
                            monitor="val_loss_epoch",
                            dirpath=model_info,
                            filename="model-{epoch:02d}-{val_loss:.2f}",
                            save_top_k=1,
                            mode="min",
                            )
    early_stopping_callback = EarlyStopping(
                            monitor = 'val_loss_epoch',
                            min_delta = 0.0, 
                            patience = 10, # after 10 epochs, just exit
                            mode = 'min'
                            )
    trainer = Trainer.from_argparse_args(args, callbacks = [checkpoint_callback, early_stopping_callback], default_root_dir = dest, max_epochs = args.num_epochs, auto_lr_find=True)

    # find optimal lr
    trainer.tune(model, train_loader, val_loader)

    # fit model
    trainer.fit(model, train_loader, val_loader)

    # load best model
    model = RFUNetTraining.load_from_checkpoint(checkpoint_callback.best_model_path, model=unet, loss_fcn=loss_fn, lr = args.lr, momentum = args.momentum, train_size = args.new_size)

    # get normal stats
    stats_dir = join(dest, 'statistics')
    os.mkdir(stats_dir)

    final_test_metrics(trainer, model, val_loader, test_loader, save_path = stats_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    

    # training info
    parser.add_argument('-mode', dest = 'mode', type = str, required = True, help = 'Mode: Train or Test')
    parser.add_argument('-model_path', dest = 'model_path', type = str, help = 'If Mode = Test, path containing a previously trained model checkpoint. Nonoperational for mode=Train')
    parser.add_argument('-data_path', dest = 'data_path', required=True, help = 'Datapath containing augmented data. Must contain a train, val, test folder which each have images, (targets), and masks')
    parser.add_argument('-save_path', dest = 'save_path', required = True, help = 'Path to save folder. If Mode = Train, Should be Nonexistent, but will created a duplicate save_path_X for X = 1-5 if it does. If Mode: Test, this folder should be the folder you would like to save statistics in.')
    parser.add_argument('-num_epochs', dest = 'num_epochs', type = int, default = 50, help = 'Number of Max epochs to run. Defaults to 50')
    parser.add_argument('-train_batch', dest = 'train_batch', type = int, default = 1, help = 'Training batch size. Defaults to 1')
    parser.add_argument('-lr', dest = 'lr', type = float, default = .001, help = 'Optimizer starting Learning Rate. However, will be optimized by Pytorch Lightning. Defaults to .001')
    parser.add_argument('-momentum', dest = 'momentum', type = float, default = .99, help = 'Momentum the Optimizer will use. Defaults to .99')
    parser.add_argument('-block_size', dest = 'block_size', type = int, default = 7, help = 'Block size of dropblock, which must be odd numbers. A size of 1 is equivalent to dropout. Defaults to 7.')
    parser.add_argument('-max_drop_prob',dest = 'max_drop_prob', type = float, default = .15, help = 'Maximum drop probability of dropblock, must be from 0-1. Defaults to .15')
    parser.add_argument('-dropblock_steps', dest = 'dropblock_steps', type = int, default = 1500, help = 'Number of steps before max drop prob is reached. Defaults to 1500')
    parser.add_argument('-new_size', dest = 'new_size', type = int, default = 32, help = 'Minimum size of the crop during training.')
    parser.add_argument('-train_ratio', dest = 'train_ratio', type = float, default = 1, help = 'Ratio of data to use while training. Defaults to 1, or all data.')
    parser.add_argument('-seed', dest = 'seed', type = int, default = -1, help = 'Seed for reproducability. Defaults to -1, which is equivalent to None' )
    parser = Trainer.add_argparse_args(parser)

    # testing info

    args = parser.parse_args()
    
    if args.mode == 'train':
        training(args)
    elif args.mode == 'test':
        testing(args)