import numpy as np
from torch._C import device
from torch.functional import split
from torch.serialization import save
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import os
from utils.utils_general import split_target, get_masked, TensortoPIL






# validation epoch
def val_epoch_debug(network, loss_fn, dataloader, device, use_mask = True):
  
    num_cols = 1
    fig, ax = plt.subplots(1, 7, figsize = (5 * 7, 5), squeeze = True)



    print('Debugging')
    with torch.no_grad():
        batch_idx, (image_batch, gt, mask)  = next(enumerate(dataloader))
        
        image_batch = image_batch.to(device)
    
        segmentation = network(image_batch)
        
        # process gt
        gt = split_target(gt)
        gt = gt.to(device)

        
        # if using masks, only get region with mask
        if use_mask:
            segmentation2, gt, mask = get_masked(segmentation, gt, mask, device)
        
            
        ax[0].imshow(TensortoPIL(image_batch.cpu()[0]))
        ax[1].imshow(TensortoPIL(segmentation.cpu()[0][0]), cmap = 'gray')
        ax[2].imshow(TensortoPIL(segmentation.cpu()[0][1]), cmap = 'gray')
        ax[3].imshow(TensortoPIL(gt.cpu()[0][0]), cmap = 'gray')
        ax[4].imshow(TensortoPIL(gt.cpu()[0][1]), cmap = 'gray')
        ax[5].imshow(TensortoPIL(mask.cpu()[0][0]), cmap = 'gray')
        ax[6].imshow(TensortoPIL(mask.cpu()[0][1]), cmap = 'gray')
        ax[0].set_title('Original Image')
        ax[1].set_title('Segmentation Class 1')
        ax[2].set_title('Segmentation Class 2')
        ax[3].set_title('GT Class 1')
        ax[4].set_title('GT Class 2')
        ax[5].set_title('Mask Class 1')
        ax[6].set_title('Mask Class 2')

        fig.suptitle(f'Image Debug')
        
        print("Segmentation: ", segmentation)
        print("Seg Max: ", segmentation.max())
        print("Seg Min: ", segmentation.min())
        print("Number of non 0's:", segmentation.count_nonzero())
        print("Number of 0's: ", segmentation.numel()-segmentation.count_nonzero())
        
        print("GT: ", gt)
        print("GT Max: ", gt.max())
        print("GT Min: ", gt.min())
        print("Number of non 0's:", gt.count_nonzero())
        print("Number of 0's: ", gt.numel()-gt.count_nonzero())
        #get orig image and segmentation
        loss = loss_fn(segmentation, gt)
        
        # multiply by total pixels and divide by number of 1 pixels in mask
        if use_mask:
            loss *= (segmentation.numel() / mask.count_nonzero())
        
        torch.cuda.empty_cache()
        fig.show()
    return loss.detach().cpu().numpy()

