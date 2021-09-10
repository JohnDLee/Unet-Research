import torch
from torch import nn
from torchvision import transforms, datasets, io

import os
from PIL import Image #image reader
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torchvision.transforms.functional as TF

from utils.utils_general import split_target


class CustomDataset(torch.utils.data.Dataset):


  def __init__(
      self,
      image_root: str,
      target_root: str = None,
      mask_root: str = None,
      transform: Optional[Callable] = None,
  ):
    ''' Assumes im_root and target_root contain images labeled numerically'''
    # images
    self._im_root = image_root
    self._im_list = sorted(os.listdir(image_root))

    self._target_root = target_root
    if target_root:
      self._target_list = sorted(os.listdir(target_root))
    
    self._mask_root = mask_root
    if mask_root:
      self._mask_list = sorted(os.listdir(mask_root))

    # transforms
    self._transform = transform

    self._data_type = 'train'
    
  def __getitem__(self, idx):
    # get image path
    img_name = self._im_list[idx]
    img_path = os.path.join(self._im_root, img_name)
    
    # open and convert image to tensor
    image = Image.open(img_path)
    image = image.convert('RGB') # 3 channels

    # retrieve labels
    if self._target_root:
      # retreive label
      label_name = self._target_list[idx]
      label_path = os.path.join(self._target_root, label_name)

      # open and convert label to tensor
      label = Image.open(label_path)
      label = label.convert('L') # 1 channel
    else:
      label = torch.zeros((1, image.size[0], image.size[1]))

    # retrieve mask
    if self._mask_root:
      mask_name = self._mask_list[idx]
      mask_path = os.path.join(self._mask_root, mask_name)
      
      mask = Image.open(mask_path)
      mask = mask.convert('L') # 1 Channel
    else:
      mask = torch.ones((1, image.size[0], image.size[1]))
      
    sample = (image, label, mask)

    if self._transform and self._data_type != 'validation':
      sample = self._transform(sample)

    return sample
  
  def __len__(self):
    return len(self._im_list)

# Transformation Functions 
# (changed to accept a tuple of (image, target) to apply identical transformations to both )



class RandomOperations:

  def __init__(self, transforms, weights = None):
    ''' transformations to perform
    weights: probability for each transformation to be performed
    '''
    self._transforms = transforms
    if weights is None:
      self._weights == [.5 for i in range(len(transforms))]
    else:
      if len(weights) == len(transforms):
        self._weights = weights
      else:
        print('Length of weights does not match length of transform operations')

  def __call__(self, tup):
    for index in range(len(self._transforms)):
      if torch.rand(1) < self._weights[index]:
        tup = self._transforms[index](tup)
    
    return tup
  
class AutoPad(nn.Module):
  
  def __init__(self, original_size, model_depth, fill = 0, padding_mode = "constant"):
    super().__init__()
    
    right_pad = 0
    while ((original_size[1] + right_pad) % (2**model_depth) != 0):
      right_pad += 1
    
    bot_pad = 0
    while ((original_size[0] + bot_pad) % (2**model_depth) != 0):
      bot_pad += 1
    
    self._pad_sequence = (0, 0, right_pad, bot_pad)
    self._fill = fill
    self._padding_mode = padding_mode
    
  def forward(self, tup):
    ''' accepts a 2D array-like containing PIL or Tensor
    return a Tuple of padded images, automatically padded to a size divisible by 2^model_depth
    '''
    
    new_tup = []
    for i in range(len(tup)):
      new_tup.append(TF.pad(tup[i], self._pad_sequence, self._fill, self._padding_mode ))
    return tuple(new_tup)
  

class Pad(nn.Module):
  
  def __init__(self, pad_sequence = (0,0,0,0), fill = 0, padding_mode = "constant"):
    super().__init__()
    self._pad_sequence = pad_sequence
    self._fill = fill
    self._padding_mode = padding_mode
    
  def forward(self, tup):
    ''' accepts a 2D array-like containing PIL or Tensor
    return a Tuple of padded images, padded left, top, right, bottom respectively accourding to sequence.
    '''
    
    new_tup = []
    for i in range(len(tup)):
      new_tup.append(TF.pad(tup[i], self._pad_sequence, self._fill, self._padding_mode ))
    return tuple(new_tup)


#hflip
class RandomHorizontalFlip(nn.Module):

  def __init__(self, p = .5):
    super().__init__()
    self._p = p
  
  def forward(self, tup):
    ''' accepts an 2D array-like containing PIL or Tensors

    returns a Tuple of randomly flipped PIL or Tensor
    '''
    if torch.rand(1) < self._p:
      
      new_tup = []
      for i in range(len(tup)):
        new_tup.append(TF.hflip(tup[i]))
      return tuple(new_tup)
    return tup
  
#vflip
class RandomVerticalFlip(nn.Module):

  def __init__(self, p = .5):
    super().__init__()
    self._p = p
  
  def forward(self, tup):
    ''' accepts an 2D array-like containing PIL or Tensors

    returns a Tuple of randomly flipped PIL or Tensor
    '''
    if torch.rand(1) < self._p:
      new_tup = []
      for i in range(len(tup)):
        new_tup.append(TF.vflip(tup[i]))
      return tuple(new_tup)
    return tup

#rotate
class RandomRotate(nn.Module):

  def __init__(self, degrees, center = None, fill = 0):
    ''' degrees (int or Tuple): degrees to randomly calculate angle for'''

    super().__init__()
    if type(degrees) == int: # integer
      self._degrees = (-degrees, degrees)
    else:
      self._degrees = degrees

    self._center = center

    if fill is None:
      fill = 0

    self._fill = fill

  def forward(self, tup):
    #get random angle
    angle = float(torch.empty(1).uniform_(float(self._degrees[0]), float(self._degrees[1])).item())
    
    new_tup = []
    for i in range(len(tup)):
      new_tup.append(TF.rotate(img = tup[i], angle = angle, center = self._center, fill = self._fill))
    return tuple(new_tup)
    
    
#Translate
class RandomTranslate(nn.Module):

  def __init__(self, translate):
    super().__init__()
    if type(translate) == int:
      translate = (translate, translate)
    self._translate = translate

  def forward(self, tup):
    
    img_size = tup[0].size()
    
    if self._translate is not None:
      max_dx = float(self._translate[0] * img_size[0])
      max_dy = float(self._translate[1] * img_size[1])
      tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
      ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
      translations = (tx, ty)
    else:
      translations = (0, 0)
      
    new_tup = []
    for i in range(len(tup)):
      new_tup.append(TF.affine(tup[i], translate = translations))
    return tuple(new_tup)
  
    
#pil_to_tensor
class ToTensor:
  def __call__(self, tup):
    new_tup = []
    for i in range(len(tup)):
      if type(tup[i]) != torch.Tensor:
        new_tup.append(TF.to_tensor(tup[i]))
      else:
        new_tup.append(tup[i])
    return tuple(new_tup)

