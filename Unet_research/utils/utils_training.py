import numpy as np
from torch._C import device
from torch.functional import split
from torch.serialization import save
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import os

from utils.utils_general import split_target, get_masked, TensortoPIL
from utils.utils_metrics import *


# training epoch
def train_epoch(epoch, network, optimizer, loss_fn, dataloader, device,use_mask = True, debug = False, verbose = False):
    
    if verbose:
        print(f"\nTrain Epoch {epoch}")
    # turn training mode on
    network.train()
    losses = []
    for batch_idx, (image_batch, gt, mask) in enumerate(dataloader):
      
        # move images to device
        image_batch = image_batch.to(device)
        if debug:
            print('After Image Batch:\n',torch.cuda.memory_summary(device))
        
        segmentation = network(image_batch)
        if debug:
            print('After Model Segmentation:\n',torch.cuda.memory_summary(device))
        
        # process gt
        gt = split_target(gt) # split the target
        gt = gt.to(device)
        if debug:
            print('After GT:\n',torch.cuda.memory_summary(device))
        
        # multiply both segmentation and gt by mask
        if use_mask:
            segmentation, gt, mask = get_masked(segmentation, gt, mask, device)
        if debug:
            print('After Mask:\n',torch.cuda.memory_summary(device))
        
        loss = loss_fn(segmentation, gt)

        # multiply by size of tensor, divide by number of pixels that are 1 in the mask
        if use_mask:
          loss *= (segmentation.numel() / mask.count_nonzero())
        if debug:
            print('After Loss Recalculation:\n',torch.cuda.memory_summary(device))

        #backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if debug:
            print('After Backpropogation:\n',torch.cuda.memory_summary(device))
            
        # get info
        #logger.info(f'\tBatch {batch_idx + 1}/{len(dataloader)}: loss - {loss}')
        losses.append(loss.item())
        
        # if verbose = True, print a message
        if verbose:
            print(f'\tBatch {batch_idx + 1}/{len(dataloader)}: loss = {losses[-1]}')
        
        del segmentation, gt, mask, loss
        
        torch.cuda.empty_cache() # clear GPU
        
        if debug:
            print('After Clean:\n', torch.cuda.memory_summary(device))
  
    
  
    return np.array(losses).mean()


# validation epoch
def val_epoch(epoch, network, loss_fn, dataloader, device, use_mask = True, verbose = False):

    if verbose:
        print(f"\nVal Epoch {epoch}")
    # set to evaluation mode
    network.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (image_batch, gt, mask) in enumerate(dataloader):
        
            image_batch = image_batch.to(device)
            
            segmentation = network(image_batch)
            
            
            # process gt
            gt = split_target(gt)
            gt = gt.to(device)

            
            # if using masks, only get region with mask
            if use_mask:
                segmentation, gt, mask = get_masked(segmentation, gt, mask, device)

            #get orig image and segmentation
            loss = loss_fn(segmentation, gt)
            
            # multiply by total pixels and divide by number of 1 pixels in mask
            if use_mask:
              loss *= (segmentation.numel() / mask.count_nonzero())
            
            losses.append(loss.item()) # save losses
            
            if verbose:
                print(f'\tBatch {batch_idx + 1}/{len(dataloader)}: validation loss = {losses[-1]}')
          
            del segmentation, gt, mask, loss
            torch.cuda.empty_cache() # clear GPU

    return np.array(losses).mean()
  




