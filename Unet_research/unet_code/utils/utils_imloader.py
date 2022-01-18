import torch
import os
from PIL import Image #image reader
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class ImLoader(torch.utils.data.Dataset):

  def __init__(
      self,
      image_root: str,
      target_root: str = None,
      mask_root: str = None,
  ):
    ''' Assumes im_root and target_root contain images labeled numerically in form imID_*'''
    # images
    self._im_root = image_root
    self._im_list = sorted(os.listdir(image_root))

    self._target_root = target_root
    if target_root:
      self._target_list = sorted(os.listdir(target_root))
    
    self._mask_root = mask_root
    if mask_root:
      self._mask_list = sorted(os.listdir(mask_root))


  def __getitem__(self, idx):
    # get image path
    img_name = self._im_list[idx]
    img_path = os.path.join(self._im_root, img_name)
    
    # open
    image = np.asarray(Image.open(img_path).convert('RGB')) # 3 channels
    label = None
    mask = None

    # retrieve labels
    if self._target_root:
      # retreive label
      label_name = self._target_list[idx]
      label_path = os.path.join(self._target_root, label_name)

      # open and convert label to tensor
      label = np.asarray(Image.open(label_path).convert('L')) # 1 channel
    
    # retrieve mask
    if self._mask_root:
      mask_name = self._mask_list[idx]
      mask_path = os.path.join(self._mask_root, mask_name)
      
      mask = np.asarray(Image.open(mask_path).convert('L')) # 1 Channel

    return image, label, mask
  
  def __len__(self):
    return len(self._im_list)
