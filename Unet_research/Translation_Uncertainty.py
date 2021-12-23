import os
import torch
import torch.nn as nn
import json
import copy
from torchvision.transforms.functional import affine
from dropblock import DropBlock2D
from torch.utils.data import DataLoader
import numpy as np

from utils.utils_unet import UNet
from utils.utils_general import split_target, split_mask
from utils.utils_dataset import *
from utils.utils_metrics import *


def sort_params(name_list, param_dict):
    '''sorts the params given a name list'''
    sectioned_params = {}
    for param in name_list:
        if param in param_dict['optimized'].keys():
            sectioned_params[param] = param_dict['optimized'][param]
        elif param in param_dict['constant'].keys():
            sectioned_params[param] = param_dict['constant'][param]
            
    return sectioned_params

class TranslateGenerator():
    ''' Sections of pieces of the data and hashes them
    to allow for reconstruction'''
    def __init__(self, image_batch, stamp_size = 32):
        self._image_batch = image_batch
        self._stamp_size = stamp_size
        self._image_split = self.split_image(image_batch, stamp_size) # split the image
        
        hash_list = [x for x in range(len(self._image_split))]
        self._pre_std_tensor = hash_list
        self._pre_avg_tensor = hash_list
        
        self._reconstructed_mean = None
        self._reconstructed_std = None
        
    def split_image(self, image_batch, size):
        '''split an image by size into an unsqueezed list'''
        temp = [image_batch[:,:,:,i:i+size] for i in range(0, image_batch.size()[3], size)]
        split_tensor = []
        for t in temp:
            split_tensor+=[t[:,:,i:i+size] for i in range(0, image_batch.size()[2], size)]
        return split_tensor
    
    def get_images(self):
        '''generator for our images and their corresponding hash'''
        hash = 0
        for split in self._image_split:
            yield split, hash
            hash += 1
    
    def add_to_std_tensor(self, stamp, hash):
        ''' add to our std tensor'''
        self._pre_std_tensor[hash] = stamp
    
    def add_to_mean_tensor(self, stamp, hash):
        ''' add to our mean tensor'''
        self._pre_avg_tensor[hash]= stamp
    
    def reconstruct_std(self):
        '''reconstructs the std tensor'''
        size = self._stamp_size
        height = len([i for i in range(0, self._image_batch.size()[2], size)])
        self._reconstructed_std = torch.cat([torch.cat(self._pre_std_tensor[i:i + height], dim = 2) for i in range(0, len(self._pre_avg_tensor), height)], dim= 3)
        return self._reconstructed_std
        
    
    def reconstruct_mean(self):
        '''reconstructs the mean tensor'''
        size = self._stamp_size
        height = len([i for i in range(0, self._image_batch.size()[2], size)])
        self._reconstructed_mean = torch.cat([torch.cat(self._pre_avg_tensor[i:i + height], dim = 2) for i in range(0, len(self._pre_avg_tensor), height)], dim= 3)
        return self._reconstructed_mean
        



def test_uncertainty( network, loss_fn, dataloader, device, num_iter = 1000, use_mask = True, verbose = False, save = True, save_location = '.'):
    ''' tests uncertainty'''
    losses = []
    with torch.no_grad():
        for batch_idx, (image_batch, gt, mask) in enumerate(dataloader):
            
            tensors = []
            
            # move image to GPU
            #image_batch = image_batch.to(device)
            # process gt
            gt = split_target(gt)
            #gt = gt.to(device)
            # process mask
            mask = split_mask(mask)
            mask = mask.to(device)
            
            for iter in range( num_iter ):
                
                #perform translation if second image
                trans_image = affine(image_batch[0], translate=(50 * iter, 0), angle =0, shear = 0, scale = 1.0)   # translate 50 pixels for each sample, 5 samples
                trans_image = torch.unsqueeze(trans_image, 0)
                
                trans_image = trans_image.to(device)
                
                segmentation = network(trans_image)
                segmentation = segmentation.cpu()
                #perform inverse translation
                
                segmentation = affine(segmentation[0], translate = (-50 * iter, 0), angle =0, shear = 0, scale = 1.0)
                segmentation = torch.unsqueeze(segmentation, 0)
                

                
                tensors.append(segmentation.detach()) # append our segmentation
                if verbose and iter % 50 == 0:
                    print(f'\tIteration {iter}: {iter}/{num_iter}')
                del segmentation
                torch.cuda.empty_cache() # clear GPU
                
            # save our results
            tensors = torch.stack(tensors)
            
            # get mean & std
            avg_segmentation = tensors.mean(0) # take pixel wise mean
            std_tensor = tensors.std(0) # take pixel wise std
            
            mask = mask.cpu()
            if use_mask:
                avg_segmentation = avg_segmentation * mask # mask again, just to be safe
                masked_gt = gt * mask
              

                
            #get orig image and segmentation
            loss = loss_fn(avg_segmentation, masked_gt)
            
            # multiply by total pixels and divide by number of 1 pixels in mask
            if use_mask:
              loss *= (avg_segmentation.numel() / mask.count_nonzero())
            
            losses.append(loss.item()) # save losses
            
            if verbose:
                print(f'\tBatch {batch_idx + 1}/{len(dataloader)}: validation loss = {losses[-1]}')
          
          
          
            ####### PROCESS OUR TENSORS #######
          
          
          
            if save:
                avg_segmentation = avg_segmentation.cpu()
                masked_gt = masked_gt.cpu()
                image_batch = image_batch.cpu()
                
                im_path = os.path.join(save_location , f'dropblock_test{batch_idx}')
                if not os.path.exists(im_path):
                    os.mkdir(im_path)
                
                #don't save tensors for this run !~!!!~~
                torch.save(tensors.clone(), os.path.join(im_path, 'tensors.pt'))
                torch.save(avg_segmentation.clone(), os.path.join(im_path, 'mean_tensor.pt'))
                torch.save(std_tensor.clone(), os.path.join(im_path, 'std_tensor.pt'))
                save_contour_map(avg_segmentation[0], masked_gt[0], save_path=im_path)
                save_example(image_batch[0], avg_segmentation[0], unbatched_gt=masked_gt[0], id = batch_idx, save_path = im_path)
                
            del tensors
            del avg_segmentation, gt, mask, loss, masked_gt
            torch.cuda.empty_cache() # clear GPU

    return np.array(losses)

    


if __name__=='__main__':
    
    results_root = 'results'
    
    if not os.path.exists(results_root + '/model.pth') or not os.path.exists(results_root + '/model_params'): # if model does not exist quit
        print('Model does not exist.')
        quit()
    
    # get parameters from json.load
    with open(results_root + '/model_params','r') as f:
        param_dict = json.load(f)

    # retrieve our UNet parameters for loading
    unet_param_names = 'activation_fcn connection block_size dropblock_ls_steps model_depth max_drop_prob neg_slope pool_mode use_batchnorm use_dropblock filters up_mode conv_layers_per_block init_channels output_channels same_padding'.split(' ')
    unet_params = sort_params(unet_param_names, param_dict)
    model_depth = sort_params(['model_depth'], param_dict)['model_depth']
    
    # initialize UNet with our params
    unet = UNet(**unet_params)
    
    
    # check if GPU is available, if not, exit
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print('GPU unavailable, CPU will take extremely long. Exitting...')
        quit()
        
    # Load our model with our pretrained model
    unet.load_state_dict(torch.load(results_root + '/model.pth',  map_location=torch.device(device)))
    
    unet.to(device) # move unet to gpu
    unet.eval() # set unet to eval mode
    
    # use validation data to determine uncertainty
    val_root = 'final_data/gray_validation/'
    if not os.path.exists(val_root):
        print('Final Input validation image folder does not exist. Exitting...')
        quit()
    
    # retrieve our data
    val_transform = transforms.Compose([AutoPad(original_size = (584, 565), model_depth = model_depth, fill = 0, padding_mode = "constant"),
                                     ToTensor()])
    val_dataset = CustomDataset(image_root=val_root + 'images',
                                  target_root=val_root + 'targets',
                                  mask_root=val_root + 'masks',
                                  transform=val_transform)
    val_batch_size = 1 # load 1 image at a time -> no problem with torch.grad = False

    # load into dataloaders
    val_loader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = False)
    
    # loss function
    loss_fn = nn.BCELoss()
    
    # if new uncertainty location does not exist, make it
    uncertainty_root = results_root + '/Translation_MonteCarlo'
    if not os.path.exists(uncertainty_root):
        os.mkdir(uncertainty_root)
        
    # test for uncertainty
    losses = test_uncertainty( network = unet,
                     loss_fn = loss_fn,
                     dataloader = val_loader,
                     device = device,
                     num_iter = 5,
                     use_mask = True,
                     verbose = True,
                     save = True,
                     save_location = uncertainty_root)
    

    # save our numpy data
    losses.tofile(os.path.join(uncertainty_root, 'avged_losses.txt'), sep = '\n', format = '%ls')
    
    
    

    
    
    
    
    
    
    
        
