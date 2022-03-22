import sys
import os
from os.path import join, exists
from cv2 import resize
import torch
from pytorch_lightning import seed_everything
import argparse
import albumentations as A
from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF
sys.path.append(os.getcwd() + '/unet_code')
from utils.utils_imloader import ImLoader
from utils.utils_preprocessing import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dest', dest = 'dest', type = str, default = 'augmented_data', help = 'Sets the location for the new data')
    parser.add_argument('-seed', dest = 'seed', type = int, default = 1234, help = 'Sets the seed for reproducability' )

    args = parser.parse_args()
    
    # set seed
    seed_everything(args.seed)

    # specify data path
    training_root = 'datasets/training'
    test_root = 'datasets/test'


    given = ImLoader(image_root=join(training_root, 'images'),
                            target_root=join(training_root, '1st_manual'),
                            mask_root=join(training_root ,'mask'))

    test = ImLoader(image_root=join(test_root, 'images'),
                            mask_root=join(test_root, 'mask'))
    
    # split given
    training_pct = .7
    training_len = int (len(given) * training_pct)

    # training and val
    training, val = torch.utils.data.random_split(given, [training_len, len(given) - training_len])


    # set up transforms
    train_transform = A.Compose([A.ToGray(always_apply = True),
                                A.Flip(p = .5), # randomly flip horizontally or vertically or both
                                A.Rotate(limit = 180, p = .95, border_mode=1),
                                #A.Resize(args.height, args.width,always_apply=True )
                                ], additional_targets = { 'image': 'image',
                                                    'target': 'mask',
                                                    'mask': 'mask'}
                                )
    val_transform = A.Compose([A.ToGray(always_apply = True),
                                #A.Resize(args.height, args.width,always_apply=True )
                                ], 
                                additional_targets = { 'image': 'image',
                                                    'target': 'mask',
                                                    'mask': 'mask'}
                                )
    test_transform = A.Compose([A.ToGray(always_apply = True),
                                #A.Resize(args.height, args.width,always_apply=True )
                                ], 
                                additional_targets = { 'image': 'image', 'mask':'mask'})


    # create destination location
    dest = args.dest
    if not exists(dest):
        os.mkdir(dest)
    else:
        # attempt 4 extra times
        for i in range(1,5):
            dest = args.dest + str(i)
            if not exists(dest):
                os.mkdir(dest)
                break
        else: # if completes normally
            print("Could not create destination directory.")
            exit(1)
    
    # create folders inside
    train_dest = join(dest, 'train')
    val_orig_dest = join(dest, 'val') # original validations
    test_dest = join(dest, 'test')

    for paths in [train_dest, val_orig_dest, test_dest]:
        os.mkdir(paths)
    
    num_train = 36
    # populate train_dest
    gen_givens(train_dest, num = num_train, loader = training, transformation=train_transform, seed = args.seed)
    
    # populate val_orig_dest
    gen_givens(val_orig_dest, num = 1, loader = val, transformation=val_transform, seed = args.seed)

    # populate test_dest
    gen_tests(test_dest, num = 1, loader = test, transformation=test_transform)


