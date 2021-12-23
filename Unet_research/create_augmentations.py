import os
import shutil
from utils.utils_augmentations import *
import albumentations as A
import numpy as np

import argparse

def train_validation_split(train_path, destination_dir, train_pct = .7, seed = None):
    '''splits the train and validation data -> stores the validation and train in destination path'''
    if seed is not None:
        np.random.seed(seed)
    
    # get train images 
    train_path_images = os.path.join(train_path, 'images')
    train_path_target = os.path.join(train_path, '1st_manual')
    train_path_mask = os.path.join(train_path, 'mask')
    
    # create subdirectories
    validation_root = os.path.join(destination_dir, 'validation')
    if not os.path.exists(validation_root):
        os.mkdir(validation_root)
        os.mkdir(os.path.join(validation_root, 'images'))
        os.mkdir(os.path.join(validation_root, 'targets'))
        os.mkdir(os.path.join(validation_root, 'masks'))
    
    train_root = os.path.join(destination_dir, 'training')
    if not os.path.exists(train_root):
        os.mkdir(train_root)
        os.mkdir(os.path.join(train_root, 'images'))
        os.mkdir(os.path.join(train_root, 'targets'))
        os.mkdir(os.path.join(train_root, 'masks'))
        
    image_list = sorted(os.listdir(train_path_images))
    target_list = sorted(os.listdir(train_path_target))
    mask_list = sorted(os.listdir(train_path_mask))
    
    # get size
    size = len(image_list)
    train_size = int(size * train_pct)

    
    # zip it all togethor
    zipped = list(zip(image_list, target_list, mask_list))
    # shuffle
    np.random.shuffle(zipped)
    
    for im, targ, mask in zipped[:train_size]:
        shutil.copy(os.path.join(train_path, 'images', im), os.path.join(train_root, 'images'))
        shutil.copy(os.path.join(train_path, '1st_manual', targ), os.path.join(train_root, 'targets'))
        shutil.copy(os.path.join(train_path, 'mask', mask), os.path.join(train_root, 'masks'))
        
    for im, targ, mask in zipped[train_size:]:
        shutil.copy(os.path.join(train_path, 'images', im), os.path.join(validation_root, 'images'))
        shutil.copy(os.path.join(train_path, '1st_manual', targ), os.path.join(validation_root, 'targets'))
        shutil.copy(os.path.join(train_path, 'mask', mask), os.path.join(validation_root, 'masks'))
    
    

    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description= "Preprocesses UNET training images", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f',dest = 'destination_dir', type = str, default = 'final_data', help = 'Destination directory of modified data')
    parser.add_argument('-width', dest = 'width', type = int, default = 565, help = 'Width of final images')
    parser.add_argument('-height', dest = 'height', type = int, default = 584, help = 'Heigh of final images')
    args = parser.parse_args()
    
    # split training and validation data
    train_path = 'datasets/training'
    
    destination_dir = args.destination_dir # our destination will be specified in the arg
    
    
    
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    train_pct = .7
    
    train_validation_split(train_path = train_path,
                           destination_dir=destination_dir,
                           train_pct=train_pct)
    
    
    # generate augmentations for training data
    new_train_path = destination_dir + '/training'
    output_path = destination_dir + '/augmented_training'
    if not os.path.exists(output_path):
        print("Creating Augmented Images")
        # grayscale everything
        transforms_train = A.Compose([A.ToGray(always_apply = True),
                                A.Flip(p = .5), # randomly flip horizontally or vertically or both
                                A.Rotate(limit = 180, p = .95, border_mode=1),
                                A.Resize(args.height, args.width,always_apply=True )
                                ], additional_targets = { 'image': 'image',
                                                    'target': 'mask',
                                                    'mask': 'mask'}
                                
                                )
        augmentations(input_dir = new_train_path,
                    output_dir = output_path,
                    num_images = 500, # will approximately create 500
                    transforms = transforms_train)
    else:
        print(f"Skipping Augmentation Creation - Output Directory {output_path} exists")
    
    # grayscale our validation
    new_val_path = destination_dir + '/validation'
    output_val_path = destination_dir + '/gray_validation'
    if not os.path.exists(output_val_path):
        print("Creating Grayscale Val Images")
        val_transforms = A.Compose([A.ToGray(always_apply = True),
                                    A.Resize(args.height, args.width,always_apply=True )], 
                            additional_targets = { 'image': 'image',
                                                    'target': 'mask',
                                                    'mask': 'mask'}
                                )
        augmentations(input_dir = new_val_path,
                  output_dir = output_val_path,
                  num_images = 'same', # keep same number
                  transforms = val_transforms,
                  )
    else:
        print(f"Skipping Validation Grayscale - Output Directory {output_val_path} already exists.")
    
    
    
    # merge our test data and rename to our convention
    
    test_path = 'datasets/test'
    destination_test_path = destination_dir + '/test'
    if not os.path.exists(destination_test_path):
        shutil.copytree(test_path, destination_test_path)
    try:
        os.rename(destination_test_path + '/mask', destination_test_path  + '/masks')
    except FileNotFoundError:
        pass
    
    
    # grayscale our test
    new_test_path = destination_dir + '/test'
    output_test_path = destination_dir + '/gray_test'
    if not os.path.exists(output_test_path):
        print("Creating Grayscale Test Images")
        test_transforms = A.Compose([A.ToGray(always_apply = True),
                                     A.Resize(args.height, args.width,always_apply=True )], 
                            additional_targets = { 'image': 'image', 'mask':'mask'})
                                            
        augmentations_test(input_dir = new_test_path,
                    output_dir = output_test_path,
                    num_images = 'same', # keep same number
                    transforms = test_transforms,
                    )
    else:
        print(f"Skipping Test Grayscale - Output Directory {output_test_path} already exists.")
    
