import sys
import os
from os.path import join, exists
from xml.etree.ElementInclude import include
import torch
import argparse
import albumentations as A
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd() + '/unet_code')
from utils.utils_imloader import ImLoader

def gen_givens(dest: str, num: int, loader: ImLoader, transformation: A.Compose):
    # populate train_dest
    im_path, target_path, mask_path = gen_subdir(dest, include_targets = True)
    num_added = 0
    pbar = tqdm(total = num, desc = f"Saving images to {dest}")
    while(num_added < num):
        for im, gt, mask in loader:
            transformed = transformation(image = im,
                                            target = gt,
                                            mask = mask)

            Image.fromarray(transformed['target']).save(join(target_path, f'{num_added}_target.png'))
            Image.fromarray(transformed['image']).save(join(im_path, f'{num_added}_image.png'))
            Image.fromarray(transformed['mask']).save(join(mask_path, f'{num_added}_mask.png'))
            num_added += 1
            pbar.update(1)

def gen_tests(dest: str, num: int, loader: ImLoader, transformation: A.Compose):
    # populate train_dest
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dest', dest = 'dest', type = str, default = 'augmented_data', help = 'Sets the location for the new data')
    parser.add_argument('-tn', dest = 'num_train', type = int, default = 500, help = 'Generates n images for training. Default 500.')
    parser.add_argument('-tv', dest = 'num_val', type = int, default = 20, help = 'Generates n images for augmented validation. Default 20.')
    parser.add_argument('-seed', dest = 'seed', type = int, default = 1234, help = 'Sets the seed for reproducability' )

    args = parser.parse_args()
    
    # set seed
    torch.manual_seed(args.seed)

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
    val_aug_dest = join(dest, 'val_aug') # augmented validations
    test_dest = join(dest, 'test')

    for paths in [train_dest, val_orig_dest, val_aug_dest, test_dest]:
        os.mkdir(paths)
    
    # populate train_dest
    gen_givens(train_dest, num = args.num_train, loader = training, transformation=train_transform)
        
    # populate val_orig_dest
    gen_givens(val_orig_dest, num = len(val), loader = val, transformation=val_transform)

    # populate val_aug_dest
    gen_givens(val_aug_dest, num = args.num_val, loader = val, transformation=train_transform)

    # populate test_dest
    gen_tests(test_dest, num = len(test), loader = test, transformation=test_transform)


