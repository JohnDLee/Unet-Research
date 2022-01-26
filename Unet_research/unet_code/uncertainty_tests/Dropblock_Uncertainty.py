import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from os.path import join, exists
from PIL import Image #image reader
import numpy as np
import argparse
from pytorch_lightning import Trainer, seed_everything
import sys
import pdb

sys.path.append(os.path.join(os.getcwd(), 'unet_code'))
from utils.utils_unet import UNet
from utils.utils_modules import DropBlock2D, Dropblock2d_ichan
from utils.utils_training import BaseUNetTraining
from utils.utils_dataset import UnetDataset
from utils.utils_general import create_dir
from utils.utils_metrics import final_test_metrics

def set_dropblock_on(layer):
    ''' sets dropblock to be in training mode '''
    if type(layer) == DropBlock2D or type(layer) == Dropblock2d_ichan:
        layer.training = True

class DropBlockEval(BaseUNetTraining):
    """ base training module for UNet, alter predict step for more predictions"""

    def __init__(self, model, num_iterations = 1000, return_num = 25, mode = 'save'):
        """return_num decides how many tensor to return during MonteCarlo Dropblock prediction per prediction. Max is num_iterations.
        Beware memory issues.
        """
        super(DropBlockEval, self).__init__(model, loss_fcn=None, optimizer = None)
        self.num_iterations = num_iterations
        if return_num > num_iterations:
            self.return_num = num_iterations
        else:
            self.return_num = return_num
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode
        assert self.mode in ['save', 'evaluate']


    def predict_step(self, batch, batch_idx):
        im, gt, mask = batch
        self._model.apply(set_dropblock_on) # turn on dropblocks
        
        # run num_iter times
        tensors = torch.vstack([(self._model(im) * mask).unsqueeze(0) for _ in range(self.num_iterations)])

        mean = tensors.mean(0)
        std = tensors.std(0)

        if self.mode == 'save':
            return batch_idx, (mean, std, tensors[0:self.return_num].clone())
        elif self.mode == 'evaluate':
            return batch_idx, mean, im, gt
        



def test_uncertainty(args):

    # seed
    if args.seed != -1:
        seed_everything(args.seed, workers = True)

    
    # create new statistics folder if exists
    stats = create_dir(args.save_path)
    if stats is None:
        exit(1)
    
    # create a symlink to the model checkpoint for reference
    os.symlink(args.model_path, join(stats, 'model_ckpt_symlink.ckpt'))


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
    # check dropblock version
    if args.independent == True:
        dropblock = Dropblock2d_ichan
    else:
        dropblock = DropBlock2D
    
    unet.set_dropblock(dropblock_class = dropblock,
                        block_size = args.block_size,
                        drop_prob=args.drop_prob,
                        use_scheduler=False)
    unet.set_normalization(nn.GroupNorm, params = {'num_groups': 32, 'num_channels':"fill"})
    unet.create_model()

    # Load Training Lightning Module 
    model = DropBlockEval.load_from_checkpoint(args.model_path, model=unet, num_iterations = args.iter_num, return_num = args.save_num, mode = 'save' )

    # call Trainer
    trainer = Trainer.from_argparse_args(args, logger = False)

    # SAVE TENSORS FIRST
    # prediction
    mc_data = trainer.predict(model, dataloaders= [val_loader])
    
    tens = join(stats, 'tensors')
    os.mkdir(tens)
    # save our predictions (Evaluation elsewhere)
    for im_id, (mean, std, tensors) in mc_data:
        # create new dirs to save tensors
        im_dir = join(tens, f'image_{im_id}')
        os.mkdir(im_dir)

        # save data
        torch.save(mean, join(im_dir, 'mean.pt'))
        torch.save(std, join(im_dir, 'std.pt'))
        torch.save(tensors, join(im_dir, 'tensors.pt'))
    
    model.set_mode('evaluate')
    # EVALUATE MEAN
    statistics = join(stats, 'statistics')
    os.mkdir(statistics)
    
    final_test_metrics(trainer, model, val_loader, test_loader, save_path = statistics, disable_test = True)
        
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training info
    parser.add_argument('-model_path', dest = 'model_path', required = True, type = str, help = 'Path containing a previously trained model checkpoint.')
    parser.add_argument('-data_path', dest = 'data_path', required=True, help = 'Datapath containing augmented data. Must contain a val, test folder which each have images, (targets), and masks')
    parser.add_argument('-save_path', dest = 'save_path', required = True, help = 'Path to save folder. Should be Nonexistent, but will created a duplicate save_path_X for X = 1-5 if it does.')
    parser.add_argument('-block_size', dest = 'block_size', type = int, default = 7, help = 'Block size of dropblock, which must be odd numbers. A size of 1 is equivalent to dropout. Defaults to 7.')
    parser.add_argument('-drop_prob',dest = 'drop_prob', type = float, default = .15, help = 'Drop probability of dropblock, must be from 0-1. Defaults to .15')
    parser.add_argument('-independent_drop', dest = 'independent', action = 'store_true', help = 'Whether to use independent or dependent dropblock implementations')
    parser.add_argument('-iter_num', dest = 'iter_num', type = int, default = 1000, help = 'Number of Iterations to run MonteCarlo Simulation for each image')
    parser.add_argument('-save_num', dest = 'save_num', type = int, default = 0, help = 'Number of tensors from MonteCarlo to save. Beware memory issues.')
    parser.add_argument('-seed', dest = 'seed', type = int, default = -1, help = 'Seed for reproducability. Defaults to -1, which is equivalent to None' )
    parser = Trainer.add_argparse_args(parser)

    # testing info

    args = parser.parse_args()

    test_uncertainty(args)
    

    
    
    
    
    
    
    
        
