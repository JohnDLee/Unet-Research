import numpy as np
from torch._C import device
from torch.functional import split
from torch.serialization import save
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import os
from utils.utils_dataset import split_target

# training epoch
def train_epoch(epoch, network, optimizer, loss_fn, dataloader, device, use_mask = True, save_location = None):

  print(f'\n\nEpoch {epoch}:')

  losses = []
  for batch_idx, (image_batch, gt, mask) in enumerate(dataloader):
    
    
    
    
    # move images to device
    image_batch = image_batch.to(device)
    
    segmentation = network(image_batch)
    
    # process gt
    gt = split_target(gt) # split the target
    gt = gt.to(device)
    
    # multiply both segmentation and gt by mask
    if use_mask:
        segmentation, gt = get_masked(segmentation, gt, mask, device)
    
    
    loss = loss_fn(segmentation, gt)

    # multiply by size of tensor, divide by number of pixels that are 1 in the mask
    loss *= (segmentation.numel() / mask.count_nonzero())

    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.empty_cache()
    # get info
    print(f'\tBatch {batch_idx + 1}/{len(dataloader.dataset)}: loss - {loss}')
    losses.append(loss.detach().cpu().numpy())

  # save model
  if epoch % 5 == 0 and save_location:
    if not os.path.isdir(os.path.join(save_location, f'epoch{epoch}')):
        os.mkdir(os.path.join(save_location, f'epoch{epoch}'))
        
    torch.save(network.state_dict(), os.path.join(save_location , f'epoch{epoch}/unet.pth')) #store state_dicts to resume training at prev state if necessary
    torch.save(optimizer.state_dict(), os.path.join(save_location , f'epoch{epoch}/optimizer.pth'))
    
  return np.array(losses).mean()


# validation epoch
def val_epoch(epoch, network, loss_fn, dataloader, device, use_mask = True):
  
    print('Validating')
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
                segmentation, gt = get_masked(segmentation, gt, mask, device)
            
            loss = loss_fn(segmentation, gt)
            
            # multiply by total pixels and divide by number of 1 pixels in mask
            loss *= (segmentation.numel() / mask.count_nonzero())
            
            losses.append(loss.detach().cpu().numpy())
            
            torch.cuda.empty_cache()
    return np.array(losses).mean()


# plotting test epoch
def plot_test_epoch(epoch, network, dataloader, num_cols = 5):
  
  # alternatively

  fig, ax = plt.subplots(3, num_cols, figsize = (2 * num_cols, 2 * 3), squeeze = False, tight_layout = True)


  toPil = transforms.ToPILImage()
  with torch.no_grad():
    for batch_idx, (image_batch, _, mask) in enumerate(dataloader):
      
      segmentation = network(image_batch)
      if batch_idx < num_cols: # if space to put image, place it

        #get orig image and segmentation
        ax[0][batch_idx].imshow(toPil(image_batch[0]))
        ax[1][batch_idx].imshow(toPil(segmentation[0][0]), cmap = 'gray')
        ax[2][batch_idx].imshow(toPil(segmentation[0][1]), cmap = 'gray')

        ax[0][batch_idx].set_title(f'Image {batch_idx + 1}')
        
      else:
        break
  
  fig.suptitle(f'Test Set - Epoch {epoch}')
  plt.show()

def get_masked(segmentation, gt, mask, device):
    
    new_mask = torch.cat([mask, mask], dim = 1)
    new_mask = new_mask.to(device)
    
    return new_mask * segmentation, new_mask * gt