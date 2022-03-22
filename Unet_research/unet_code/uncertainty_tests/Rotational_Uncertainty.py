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

class RotationEval(BaseUNetTraining):
    """ base training module for UNet, alter predict step for more predictions"""

    def __init__(self, model, num_iterations = 1000, return_num = 25, resize = -1):
        """return_num decides how many tensor to return during MonteCarlo Dropblock prediction per prediction. Max is num_iterations.
        Beware memory issues.
        """
        super(RotationEval, self).__init__(model, loss_fcn=None, optimizer = None)
        self.num_iterations = num_iterations
        if return_num > num_iterations:
            self.return_num = num_iterations
        else:
            self.return_num = return_num
        self.resize = resize

    def predict_step(self, batch, batch_idx):
        im, gt, mask = batch
        
        if self.resize != -1:
            # pad to square before resizing to 
            im = square_pad(im)
            gt = square_pad(gt)
            mask = square_pad(mask)

            # perform resize on the fly on all data
            im = TF.resize(im, size = (self.resize, self.resize))
            gt = TF.resize(gt, size = (self.resize, self.resize))
            mask = TF.resize(mask, size = (self.resize, self.resize))
            
        runs = []
        for iter in range(1, self.num_iterations+1):

            # perform rotation transformation (assumes singular batch)
            rot_image = TF.rotate(im, angle = iter, interpolation=TF.InterpolationMode.BILINEAR, fill = 0)

            segmentation = self._model(rot_image)

            segmentation = TF.rotate(segmentation, angle = -iter, interpolation = TF.InterpolationMode.BILINEAR, fill = 0)
            runs.append((segmentation * mask).unsqueeze(0))
            del rot_image
        
        # run num_iter times
        tensors = torch.vstack(runs)

        mean = tensors.mean(0)
        std = tensors.std(0)

        return batch_idx, (mean, std, tensors[0:self.return_num].clone())
        



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

    val_batch_size = 1

    # load into dataloaders
    val_loader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = False, num_workers=os.cpu_count())

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
    # no dropblock. doesn't matter because it will be empty
    unet.set_normalization(nn.GroupNorm, params = {'num_groups': 32, 'num_channels':"fill"})
    unet.create_model()

    # Load Training Lightning Module 
    model = RotationEval.load_from_checkpoint(args.model_path, model=unet, num_iterations = 359, return_num = args.save_num, resize = args.resize )

    # call Trainer
    trainer = Trainer.from_argparse_args(args, logger = False)

    # prediction
    mc_data = trainer.predict(model, dataloaders= [val_loader])
    
    # save our predictions (Evaluation elsewhere)
    for im_id, (mean, std, tensors) in mc_data:
        # create new dirs to save tensors
        im_dir = join(stats, f'image_{im_id}')
        os.mkdir(im_dir)

        # save data
        torch.save(mean, join(im_dir, 'mean.pt'))
        torch.save(std, join(im_dir, 'std.pt'))
        torch.save(tensors, join(im_dir, 'tensors.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training info
    parser.add_argument('-model_path', dest = 'model_path', required = True, type = str, help = 'Path containing a previously trained model checkpoint.')
    parser.add_argument('-data_path', dest = 'data_path', required=True, help = 'Datapath containing augmented data. Must contain a val, test folder which each have images, (targets), and masks')
    parser.add_argument('-save_path', dest = 'save_path', required = True, help = 'Path to save folder. Should be Nonexistent, but will created a duplicate save_path_X for X = 1-5 if it does.')
    parser.add_argument('-save_num', dest = 'save_num', type = int, default = 0, help = 'Number of tensors from MonteCarlo to save. Beware memory issues.')
    parser.add_argument('-resize', dest = 'resize', type = int, default = -1, help =  'Resize the image before MonteCarlo.')
    parser.add_argument('-seed', dest = 'seed', type = int, default = -1, help = 'Seed for reproducability. Defaults to -1, which is equivalent to None' )
    parser = Trainer.add_argparse_args(parser)

    # testing info

    args = parser.parse_args()

    test_uncertainty(args)
    

    
    
    
    
    
    
    
        
