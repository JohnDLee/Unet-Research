import torch
from torchvision import transforms


def split_target(target):
    ''' splits our gt into [class0, class1] by inversing'''
    return torch.cat([target, 1-target], dim = 1)

def split_mask(mask):
    return torch.cat([mask, mask], dim = 1)

def get_masked(segmentation, gt, mask, device):
    ''' creates a mask for our data'''
    new_mask = torch.cat([mask, mask], dim = 1)

    new_mask = new_mask.to(device)
    
    return new_mask * segmentation, new_mask * gt, new_mask


def toPIL(tensor, mode = None):
    ''' takes a tensor of [C, W, H] and converts it to PIL Image'''
    topil = transforms.ToPILImage(mode)
    return topil(tensor)
   
    

    
    