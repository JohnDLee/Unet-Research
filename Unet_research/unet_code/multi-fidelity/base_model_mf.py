import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from os.path import join, exists
from PIL import Image #image reader
import numpy as np
import argparse
import torchvision.transforms.functional as TF
from pytorch_lightning import Trainer, seed_everything
import sys

sys.path.append(os.path.join(os.getcwd(), 'unet_code'))
from utils.utils_unet import UNet
from utils.utils_modules import DropBlock2D, Dropblock2d_ichan
from utils.utils_training import BaseUNetTraining
from utils.utils_metrics import final_test_metrics
from utils.utils_dataset import UnetDataset
from utils.utils_general import create_dir, toPIL, square_pad

class ResizeEval(BaseUNetTraining):
    """ base training module for UNet, alter predict step for more predictions"""

    def __init__(self, model, height=128, width=128):
        """return_num decides how many tensor to return during MonteCarlo Dropblock prediction per prediction. Max is num_iterations.
        Beware memory issues.
        """
        super(ResizeEval, self).__init__(model, loss_fcn=None, optimizer = None)
        self.height = height
        self.width= width

    def predict_step(self, batch, batch_idx):
        im, gt, mask = batch

        # Pad Image First to a square
        # perform resize on the fly
        im = square_pad(im)
        gt = square_pad(gt)
        mask = square_pad(mask)
        resize_image = TF.resize(im, size = (self.height, self.width))
        resize_gt = TF.resize(gt, size = (self.height, self.width))
        resize_mask = TF.resize(mask, size = (self.height, self.width))
        segmentation = self._model(resize_image)
        segmentation = segmentation * resize_mask
        
        return batch_idx, segmentation, resize_image, resize_gt
        



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
    # default dropblock. doesn't matter because it will not be used
    unet.set_dropblock(DropBlock2D,
                    block_size =7,
                    drop_prob=.15,
                    use_scheduler=True,
                    start_drop_prob=0,
                    max_drop_prob=.15,
                    dropblock_ls_steps=1500)
    unet.set_normalization(nn.GroupNorm, params = {'num_groups': 32, 'num_channels':"fill"})
    unet.create_model()

    # Load Training Lightning Module 
    model = ResizeEval.load_from_checkpoint(args.model_path, model=unet, height = args.height, width = args.width )

    # call Trainer
    trainer = Trainer.from_argparse_args(args, logger = False)

    final_test_metrics(trainer, model, val_loader, test_loader, save_path = stats, disable_test=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training info
    parser.add_argument('-model_path', dest = 'model_path', required = True, type = str, help = 'Path containing a previously trained model checkpoint.')
    parser.add_argument('-data_path', dest = 'data_path', required=True, help = 'Datapath containing augmented data. Must contain a val, test folder which each have images, (targets), and masks')
    parser.add_argument('-save_path', dest = 'save_path', required = True, help = 'Path to save folder. Should be Nonexistent, but will created a duplicate save_path_X for X = 1-5 if it does.')
    parser.add_argument('-height', dest = 'height', type = int, default = 585, help = 'height to resize images to on the fly. Default is 585 (original size of Drive Dataset)')
    parser.add_argument('-width', dest = 'width', type = int, default = 564, help = 'width to resize images to on the fly. Default is 564 (original size of Drive Dataset)')
    parser.add_argument('-seed', dest = 'seed', type = int, default = -1, help = 'Seed for reproducability. Defaults to -1, which is equivalent to None' )
    parser = Trainer.add_argparse_args(parser)

    # testing info

    args = parser.parse_args()

    test_uncertainty(args)
    

    
    
    
    
    
    
    
        
