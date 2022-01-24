import torch
from torchvision import transforms
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