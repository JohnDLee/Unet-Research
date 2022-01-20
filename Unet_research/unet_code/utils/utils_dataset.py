import torch
from torchvision import transforms

import os
from PIL import Image #image reader


class UnetDataset(torch.utils.data.Dataset):


  def __init__(
      self,
      image_root: str,
      target_root: str = None,
      mask_root: str = None,
      mode: dict = None,
  ):
    ''' Assumes im_root and target_root contain images labeled numerically,
    convert image_root, target_root and mask_root images to specified mode
    mode is default in form {'image': 'L', 'target': 'L', 'mask': 'L'}'''
    # images
    self._im_root = image_root
    self._im_list = sorted(os.listdir(image_root))

    self._target_root = target_root
    if target_root:
      self._target_list = sorted(os.listdir(target_root))
    
    self._mask_root = mask_root
    if mask_root:
      self._mask_list = sorted(os.listdir(mask_root))

    # convert mode
    if mode is None:
      self._mode = {'image': 'L', 'mask': 'L', 'target': 'L'}
    else:
      self._mode = mode
    
  def __getitem__(self, idx):
    # get image path
    img_name = self._im_list[idx]
    img_path = os.path.join(self._im_root, img_name)
    
    # open and convert image to tensor
    image = Image.open(img_path)
    image = image.convert(self._mode['image']) # 1 channels
    image = transforms.ToTensor()(image)
    # retrieve labels
    if self._target_root:
      # retreive label
      label_name = self._target_list[idx]
      label_path = os.path.join(self._target_root, label_name)

      # open and convert label to tensor
      label = Image.open(label_path)
      label = label.convert(self._mode['target']) # 1 channel
      label = transforms.ToTensor()(label)
    else:
      label = torch.zeros((1, image.shape[1], image.shape[2]))

    # retrieve mask
    if self._mask_root:
      mask_name = self._mask_list[idx]
      mask_path = os.path.join(self._mask_root, mask_name)
      
      mask = Image.open(mask_path)
      mask = mask.convert(self._mode['mask']) # 1 Channel
      mask = transforms.ToTensor()(mask)

    else:
      mask = torch.ones((1, image.shape[1], image.shape[2]))
      
    sample = (image, label, mask)

    return sample
  
  def __len__(self):
    return len(self._im_list)
