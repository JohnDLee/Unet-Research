import albumentations as A
from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm

def augmentations(input_dir, output_dir, num_images, transforms):
    ''' takes the input path to training, target, and mask '''
    
    
    #paths
    im_path = os.path.join(input_dir, 'images')
    target_path = os.path.join(input_dir, 'targets')
    mask_path = os.path.join(input_dir, 'masks')
    
    # images
    im_list = sorted(os.listdir(im_path))

    target_list = sorted(os.listdir(target_path))
    
    mask_list = sorted(os.listdir(mask_path))
    
    
    
    
    # check if files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    images_out = os.path.join(output_dir, 'images')
    targets_out = os.path.join(output_dir, 'targets')
    masks_out = os.path.join(output_dir, 'masks')
    if not os.path.exists(images_out):
        os.mkdir(images_out)
    if not os.path.exists(targets_out):
        os.mkdir(targets_out)
    if not os.path.exists(masks_out):
        os.mkdir(masks_out)
        
    # get num per image
    if num_images == 'same':
        num_per_image = 1
    else:
        num_per_image = num_images // len(im_list)
    
    
    for im_id in tqdm(range(len(im_list)), desc = 'Saving images'):

        target = np.asarray(Image.open(os.path.join(target_path, target_list[im_id] )).convert("L"))
        image = np.asarray(Image.open(os.path.join(im_path, im_list[im_id] )).convert("RGB"))
        mask = np.asarray(Image.open(os.path.join(mask_path, mask_list[im_id] )).convert("L"))
        
        for i in range(num_per_image):
            save_id = i + 1 + (im_id *  num_per_image)
            
            transformed = transforms(image = image, target = target, mask = mask)
            
            Image.fromarray(transformed['target']).save(os.path.join(targets_out, f'{save_id}_target.png'))
            Image.fromarray(transformed['image']).save(os.path.join(images_out, f'{save_id}_image.png'))
            Image.fromarray(transformed['mask']).save(os.path.join(masks_out, f'{save_id}_mask.png'))
            

            
def augmentations_test(input_dir, output_dir, num_images, transforms):
    ''' takes the input path to training, target, and mask '''
    
    
    #paths
    im_path = os.path.join(input_dir, 'images')
    mask_path = os.path.join(input_dir, 'masks')
    
    # images
    im_list = sorted(os.listdir(im_path))
    mask_list = sorted(os.listdir(mask_path))
    
    
    
    
    # check if files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    images_out = os.path.join(output_dir, 'images')
    masks_out = os.path.join(output_dir, 'masks')
    if not os.path.exists(images_out):
        os.mkdir(images_out)
    if not os.path.exists(masks_out):
        os.mkdir(masks_out)
        
    # get num per image
    if num_images == 'same':
        num_per_image = 1
    else:
        num_per_image = num_images // len(im_list)
    
    
    for im_id in tqdm(range(len(im_list)), desc = 'Saving images'):

        image = np.asarray(Image.open(os.path.join(im_path, im_list[im_id] )).convert("RGB"))
        mask = np.asarray(Image.open(os.path.join(mask_path, mask_list[im_id] )).convert("L"))
        
        for i in range(num_per_image):
            save_id = im_list[im_id].split('_')[0]
            
            transformed = transforms(image = image, mask = mask)
            
            Image.fromarray(transformed['image']).save(os.path.join(images_out, f'{save_id}_image.png'))
            Image.fromarray(transformed['mask']).save(os.path.join(masks_out, f'{save_id}_mask.png'))
            