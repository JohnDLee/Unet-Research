
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
from utils.utils_imloader import ImLoader

def gen_givens(dest: str, num: int, loader: ImLoader, transformation: A.Compose, seed: int):
    # populate train_dest
    
    seed_everything(seed)
    
    im_path, target_path, mask_path = gen_subdir(dest, include_targets = True)
    num_added = 0
    pbar = tqdm(total = len(loader) * num, desc = f"Saving images to {dest}")
    for im, gt, mask in loader:
        for i in range(num):
            transformed = transformation(image = im,
                                            target = gt,
                                            mask = mask)

            Image.fromarray(transformed['target']).save(join(target_path, f'{num_added}_target.png'))
            Image.fromarray(transformed['image']).save(join(im_path, f'{num_added}_image.png'))
            Image.fromarray(transformed['mask']).save(join(mask_path, f'{num_added}_mask.png'))
            num_added += 1
            pbar.update(1)
            
def gen_givens_resized(dest: str, sizes: list, num: list, loader: ImLoader, transformation: A.Compose, resize_up: bool, seed: int):
    # populate train_dest
    
    # check sizes corresponds to num
    assert len(sizes) == len(num)
    
    # compute a random set from sizes
    s = []
    for size in range(len(sizes)):
        s += [sizes[size] for i in range(num[size])]
    s = np.array(s)
    np.random.shuffle(s)
    
    
    seed_everything(seed) # reset seed
    
    im_path, target_path, mask_path = gen_subdir(dest, include_targets = True)
    num_added = 0
    pbar = tqdm(total = num, desc = f"Saving images to {dest}")
    while(num_added < num):
        im_num = 0
        for im, gt, mask in loader:
            transformed = transformation(image = im,
                                            target = gt,
                                            mask = mask)
            
            if s[im_num] == -1:
                Image.fromarray(transformed['target']).save(join(target_path, f'{num_added}_target.png'))
                Image.fromarray(transformed['image']).save(join(im_path, f'{num_added}_image.png'))
                Image.fromarray(transformed['mask']).save(join(mask_path, f'{num_added}_mask.png'))
            else:
                old_size = (transformed['target'].shape[-2], transformed['target'].shape[-1])
                if resize_up:
                    TF.resize(TF.resize(Image.fromarray(transformed['target']), (s[im_num], s[im_num])), old_size).save(join(target_path, f'{num_added}_target.png'))
                    TF.resize(TF.resize(Image.fromarray(transformed['image']), (s[im_num], s[im_num])), old_size).save(join(im_path, f'{num_added}_image.png'))
                    TF.resize(TF.resize(Image.fromarray(transformed['mask']), (s[im_num], s[im_num])), old_size).save(join(mask_path, f'{num_added}_mask.png'))
                else:
                    
                    TF.resize(Image.fromarray(transformed['target']), (s[im_num], s[im_num])).save(join(target_path, f'{num_added}_target.png'))
                    TF.resize(Image.fromarray(transformed['image']), (s[im_num], s[im_num])).save(join(im_path, f'{num_added}_image.png'))
                    TF.resize(Image.fromarray(transformed['mask']), (s[im_num], s[im_num])).save(join(mask_path, f'{num_added}_mask.png'))
            num_added += 1
            pbar.update(1)
            im_num += 1
        

def gen_tests(dest: str, num: int, loader: ImLoader, transformation: A.Compose):
    # populate test_dest
    im_path, mask_path = gen_subdir(dest, include_targets = False)
    count = 1
    pbar = tqdm(total = num, desc = f"Saving images to {dest}")
    for im, gt, mask in loader:

        transformed = transformation(image = im, # should be basically an identity
                                        mask = mask)

        Image.fromarray(transformed['image']).save(join(im_path, f'{str(count).zfill(2)}_image.png'))
        Image.fromarray(transformed['mask']).save(join(mask_path, f'{str(count).zfill(2)}_mask.png'))
        count+= 1
        pbar.update(1)
        

def gen_subdir(path: str, include_targets: bool = True):
    """generates subdirs image, targets, masks"""
    im_path = join(path, 'images')
    mask_path = join(path, 'masks')
    os.mkdir(im_path)
    os.mkdir(mask_path)
    if include_targets:      
        target_path = join(path, 'targets')
        os.mkdir(target_path)
        return im_path, target_path, mask_path
    return im_path, mask_path