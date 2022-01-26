import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
from os.path import join, exists



def toPIL(tensor, mode = None):
    ''' takes a tensor of [C, W, H] and converts it to PIL Image'''
    topil = transforms.ToPILImage(mode)
    return topil(tensor)
   

def create_dir(path):
    ''' tries to create a directory 5 times otherwise fails'''
    dir = path
    if not exists(dir):
        os.mkdir(dir)
    else:
        for i in range(6):
            dir = path + str(i)
            if not exists(dir):
                os.mkdir(dir)
                break
        else:
            print("Could not create directory.")
            return None
    
    return dir

def square_pad(tensor):
    """ pads to max(H,W) so that resulting tensor has H'=W'"""

    size = max(tensor.shape[-2], tensor.shape[-1])
    total_pad = (size - tensor.shape[-2])
    top = total_pad // 2
    bot = total_pad - top
    total_pad = (size - tensor.shape[-1])
    right = total_pad // 2
    left = total_pad - right

    return TF.pad(tensor, padding = (left, top, right, bot), fill = 0)
