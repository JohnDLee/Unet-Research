from dropblock import LinearScheduler
import torch.nn as nn
import torch
import torch.nn.functional as F
from numpy import product


# Credits to https://github.com/Eliza-and-black/dropblock/blob/master/dropblock/dropblock.py, which is an implementation consistent with the original paper
# The Dropblock2D from the dropblock package was inconsistent.
class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
       
    Changes in this folk(only in DropBlock2D):
        1.Make gamma consistent with the original paper, and correlatively let the center of mask stay in the shaded green region of the original paper;
        2.Make each feature channel has its DropBlock mask as the original paper propose.
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask_center = (torch.rand(x.shape[0], x.shape[1], x.shape[2] - self.block_size + 1, x.shape[3] - self.block_size + 1) < gamma).float()
            
            mask = (nn.ZeroPad2d(self.block_size // 2))(mask_center)
            if self.block_size % 2 == 0:
                mask = mask[:, :, :-1, :-1]
                
            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob * x.shape[2] * x.shape[3] / ((self.block_size ** 2) * (x.shape[2] - self.block_size + 1) * (x.shape[3] - self.block_size + 1))
    

# channels are dropped independently in this implementation
class Dropblock2d_ichan(nn.Module):
  #Credits to Lauren Partin @ the University of Notre Dame
  def __init__(self,drop_prob=0.5, block_size=3):
    super(Dropblock2d_ichan, self).__init__()
    self.drop_prob = drop_prob
    self.block_size = block_size
  # drop-prob is the probability of zeroing a unit 
  # in paper, they use a linear scheme to decrease keep_prob as training progresses
  # start with drop-prob=0, and gradually increase to drop-prob = 0.25 (e.g.)
  def set_drop_prob(self, p):
    self.drop_prob = p

  def get_gamma(self,feat_x, feat_y):
    keep_prob = 1 - self.drop_prob
    gamma = (1-keep_prob)/(self.block_size**2) * (feat_x * feat_y)/((feat_x-self.block_size+1)*(feat_y-self.block_size+1))
    #return gamma
    return min(gamma, 1)

  def get_drop_prob(self):
    return self.drop_prob

  def forward(self, tensor):
    if not self.training or self.drop_prob == 0.:
      return tensor
    feat_x = tensor.size()[2]
    feat_y = tensor.size()[3]
    gamma = self.get_gamma(feat_x, feat_y)
    #p_tensor = torch.new_full(tensor.shape, gamma, device=tensor.device) 
    p_tensor = torch.ones_like(tensor, device=tensor.device) * gamma
    mask = torch.bernoulli(p_tensor)
    bernoulli_mask = mask.clone()
    exclude = self.block_size // 2
    mask[:,:,:exclude] = 0
    mask[:,:,:,:exclude] = 0
    mask[:,:,(feat_x-exclude):] = 0
    mask[:,:,:,(feat_y-exclude):] = 0
    # apply kernel to set block_size of neighbors to 1 around 1 values in bernoulli mask 
    data_shp = mask.size()
    pad = self.block_size // 2
    maxpool = F.max_pool2d(mask.view(-1, 1, data_shp[2], data_shp[3]),
                            kernel_size=(self.block_size, self.block_size),
                            stride=(1, 1), padding=pad)
    maxpool = maxpool.view(data_shp[0], data_shp[1], data_shp[2], data_shp[3])  

    mask = 1 - maxpool 
    tensor *= mask
      
    # if no elements are dropped, this would result in nan elements
    total_elems = product(mask.shape)
    scale_denominator = 1. - torch.true_divide(total_elems - torch.sum(mask), total_elems)
    if scale_denominator != 0:
      scaling_factor = 1. / scale_denominator  
      tensor *= scaling_factor
    return tensor


    