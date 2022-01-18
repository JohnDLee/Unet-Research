import torch
from torch import nn 
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from dropblock import DropBlock2D, LinearScheduler
import torch.nn.functional as F
from numpy import product
import os



# channels are dropped independently in this implementation
class Dropblock2d_ichan(nn.Module):
    ''' Credits to Lauren Partin,@ University of N.D.'''
    def __init__(self,drop_prob=0.5, block_size=3):
        super(Dropblock2d_ichan, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
    # drop-prob is the probability of zeroing a unit 
    # in paper, they use a linear scheme to decrease keep_prob as training progresses
    # start with drop-prob=0, and gradually increase to drop-prob = 0.25 (e.g.)
    def set_drop_prob(self, p):
        self.drop_prob = p
        
    def get_gamma(self,feat_size):
        keep_prob = 1 - self.drop_prob
        gamma = (1-keep_prob)/(self.block_size**2) * (feat_size**2)/((feat_size-self.block_size+1)**2)
        return gamma
    
    def get_drop_prob(self):
        return self.drop_prob
    
    def forward(self, tensor):

        if not self.training or self.drop_prob == 0.:
            return tensor

        feat_size = tensor.size()[1]
        gamma = self.get_gamma(feat_size)
        #p_tensor = torch.new_full(tensor.shape, gamma, device=tensor.device) 
        p_tensor = torch.ones_like(tensor, device=tensor.device) * gamma
        mask = torch.bernoulli(p_tensor)
        bernoulli_mask = mask.clone()
        kernel = torch.ones((1,1,self.block_size,self.block_size), dtype=torch.double)
        # apply kernel to set block_size of neighbors to 1 around 1 values in bernoulli mask 
        data_shp = mask.size()
        
        maxpool = F.max_pool2d(mask.view(-1, 1, data_shp[2], data_shp[3]),
                                kernel_size=(self.block_size, self.block_size),
                                stride=(1, 1), padding=self.block_size // 2).view(data_shp[0], data_shp[1], data_shp[2], data_shp[3])
        
        mask = 1 - maxpool 
        tensor *= mask

        # if no elements are dropped, this would result in nan elements
        total_elems = product(mask.shape)
        scale_denominator = 1. - torch.true_divide(total_elems - torch.sum(mask), total_elems)
        if scale_denominator != 0:
            scaling_factor = 1. / scale_denominator  
        tensor *= scaling_factor
        return tensor

class UNet(nn.Module):
# can convert this to the architecture and move the actual model creation elsewhere

    def __init__(self, init_channels, filters, output_channels, model_depth = 4, pool_mode = 'max', up_mode = 'upconv', connection = 'cat',  same_padding = True, use_batchnorm = True, use_dropblock = True, block_size = 7, max_drop_prob = .1, dropblock_ls_steps = 500, dropblock_ichan = False, conv_layers_per_block = 2, activation_fcn = 'relu', neg_slope = .01,):
        """Architecure for a UNET (requires input image dimensions to be factorable by model_depth * 2)

          inputs:
          init_channels (int): number channels original image contains (typically 1 or 3)
          filters (int): number of filters to begin with (all filters for layers following will be multiples of this)
          pool_mode (str): 'max', 'avg', 'conv' are 3 different options for the pooling methodology (max & avg result in upsampling while conv results in ConvTranspose in the decoder)
          up_mode (str): 'upsample', 'upconv' chooses between upsampling or upwards convolution
          connection (str): 'add', 'cat', 'none' are 3 options for handling 
          same_padding (bool): option to keep size same after each conv layer
          use_batchnorm (bool): include batchnorm layers
          conv_layers_per_block (int>1): number of layers per block
          activation_fcn (str): activation function of relu or leaky relu
          neg_slope (float): negative slope of leaky relu (does not affect relu)"""

        super(UNet, self).__init__()

        # check for appropriate inputs
        
        

        # constants
        self._kernel_size = 3 # should we make this variable
        self._stride = 1
        self._pool_kernel = 2
        self._pool_stride = 2
        self._upsample_factor = 2
        self._bn_momentum = .1 ** (1/2)# default -> we need to squarert this for checkpointing

        # settings
        self.same_padding = same_padding
        self._init_channels = init_channels
        self._filters = filters
        self._output_channels = output_channels
        
        if connection not in ['add', 'cat', 'none' ]: # how the skip connection is done
            print('Connection type must be of (add, cat, none). Defaulting to cat')
            connection = 'cat'
        self._connection = connection

        if same_padding:
            self._padding = 'same'
        else:
            self._padding = 0

        self._use_batchnorm = use_batchnorm
        self._bias = True
        if use_batchnorm == False:
            self._bias = False
        
        self._use_dropblock = use_dropblock
        if use_dropblock:   
            if dropblock_ichan:
                self._dropblock = LinearScheduler(Dropblock2d_ichan(block_size = block_size, drop_prob = 0.),
                                              start_value = 0.,
                                              stop_value = max_drop_prob,
                                              nr_steps = dropblock_ls_steps)
            else:
                self._dropblock = LinearScheduler(DropBlock2D(block_size = block_size, drop_prob = 0.),
                                              start_value = 0.,
                                              stop_value = max_drop_prob,
                                              nr_steps = dropblock_ls_steps)
        
        
        if pool_mode not in ['max', 'avg', 'conv']: #pooling mode
            print('Pool Mode must be of (max, avg, conv). Defaulting to max')
            pool_mode = 'max'
        self._pool_mode = pool_mode
        if up_mode not in ['upsample', 'upconv']:
            print('Up_Mode must be of (upsample, upconv). Defaulting to upconv')
            up_mode = 'upconv'
        self._up_mode = up_mode
        
        if conv_layers_per_block <= 1: # number of convolutions in each block
            print('Convolutional Layers in each block must be 2 or more. Defaulting to 2')
            conv_layers_per_block = 2
        self._conv_layers_per_block = conv_layers_per_block
        
        
        
        if activation_fcn == 'relu': # activation functions
            self._act_fcn = nn.ReLU()
        elif activation_fcn == 'leaky_relu':
            self._act_fcn = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            print('Activation Function at each step must be relu or leaky_relu. Defaulting to relu')
            self._act_fcn = nn.ReLU()
        
        self._model_depth = model_depth
        
        
        # actual network
        
        # add downblocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(nn.ModuleList(self.get_down_block(init=True))) # add first downblock (unique starting condition)
        for i in range(model_depth-1):
            self.down_blocks.append(nn.ModuleList(self.get_down_block()))
        
        # add connection blocks
        self.conn_block = self.enc_dec_conn_block() # connection from encoder to decoder (dropout?)
        
        # add upblocks
        self.up_blocks = nn.ModuleList()
        for i in range(model_depth):
            self.up_blocks.append(nn.ModuleList(self.get_up_block()))
        
        # add output convolutions
        self.output_conv = self.end_conv()


    def enc_dec_conn_block(self):
        ''' Connection block between down block and upblock'''
        layers = []
        
        layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters * 2,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
        
        self._filters *= 2
        
        
        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters, momentum = self._bn_momentum))
        
        if self._use_dropblock:
            layers.append(self._dropblock)
            
        layers.append(self._act_fcn)
        
        
        
        # additional convolutional layers in middle of upblock
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
            
            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters, momentum = self._bn_momentum))
                
            if self._use_dropblock:
                layers.append(self._dropblock)
                

                
            layers.append(self._act_fcn)

            
        
        return nn.Sequential(*layers)
        
        
    def get_down_block(self, init = False):
        '''creates a down block'''
        layers = []

        #first convolutional layer in down block
        if init == True:
            layers.append(nn.Conv2d(in_channels=self._init_channels,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
          # no change to filter amount
        else:
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters*2,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
            self._filters = self._filters*2 # change the filter amount
            
        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters, momentum = self._bn_momentum))
        
        if self._use_dropblock:
            layers.append(self._dropblock)
            
        layers.append(self._act_fcn)


        # additional convolutional layers to the down block
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))

            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters, momentum = self._bn_momentum))
            
            
            if self._use_dropblock:
                layers.append(self._dropblock)
            
                
            layers.append(self._act_fcn)
        
        
        pooling = []
        # pooling step
        if self._pool_mode == 'max':
            # uses Maxpool
            pooling.append(nn.MaxPool2d(kernel_size=self._pool_kernel,
                                     stride=self._pool_stride,))
        elif self._pool_mode == 'avg':
            # uses Avgpool
            pooling.append(nn.AvgPool2d(kernel_size=self._pool_kernel,
                                     stride=self._pool_stride))
        elif self._pool_mode == 'conv':
            # uses a conv with stride of 2
            pooling.append(nn.Conv2d(in_channels=self._filters,
                                    out_channels=self._filters,
                                  kernel_size=self._pool_kernel,
                                  stride=self._pool_stride,
                                  bias = self._bias))
            

        if self._use_batchnorm:
            pooling.append(nn.BatchNorm2d(self._filters, momentum = self._bn_momentum))
            
        if self._pool_mode == 'conv': # use relu activation before
            pooling.append(self._act_fcn) # uses relu layers if down sample is a convolution
            
        return nn.Sequential(*layers), nn.Sequential(*pooling)

    
    def get_up_block(self):
        '''reverse architecture of downblock'''
        upsample = []
        
        # Upsample Layer
        # Upsample or deconv layer depending on pooling mode
        if self._up_mode == 'upsample': # upsamples, then convolutes once to reduce filter size
            # upsample
            upsample.append(nn.Upsample(scale_factor=self._upsample_factor))
            upsample.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters//2,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
        elif self._up_mode == 'upconv':
            # conv transpose
            upsample.append(nn.ConvTranspose2d(in_channels=self._filters,
                                           out_channels=self._filters//2,
                                           kernel_size=self._pool_kernel,
                                           stride=self._pool_stride,
                                           bias = self._bias))
            
        self._filters //= 2
            
        if self._use_batchnorm:
            upsample.append(nn.BatchNorm2d(self._filters, momentum = self._bn_momentum))
            
        upsample.append(self._act_fcn)
        
        
        
        layers = []
        # First convolution after skip connection
        if self._connection in ['none', 'add']:
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
        elif self._connection == 'cat':
            # concatnation connection (not sure if i built it wrong)
            layers.append(nn.Conv2d(in_channels=self._filters*2,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
        
        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters, momentum = self._bn_momentum))
            
        if self._use_dropblock:
            layers.append(self._dropblock)
            
        layers.append(self._act_fcn)
        
        # additional convolutional layers in middle of upblock
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
            
            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters, momentum = self._bn_momentum))
                
            if self._use_dropblock:
                layers.append(self._dropblock)
                
            layers.append(self._act_fcn)
            
        return nn.Sequential(*upsample), nn.Sequential(*layers)
        
        
    def skip_connection(self, x, conn):
        ''' outputs x depending on the connection method'''

        crop = transforms.CenterCrop([x.size()[-2], x.size()[-1]]) # crop the image if necessary (if same_padding = False)

        if self._connection == 'cat':
            x = torch.cat([x,crop(conn)], dim = 1)
            
            if self._use_dropblock:
                x = self._dropblock(x)
            return x
        elif self._connection == 'add':
            x = x + crop(conn)
            
            if self._use_dropblock:
                x = self._dropblock(x)
                
            return x
        elif self._connection == 'none':
            return x


    def end_conv(self):
        
        layers = []
        
        layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._output_channels,
                                  kernel_size=1,
                                  stride=self._stride,
                                  padding=self._padding,
                                  bias = self._bias))
        
        if self._use_dropblock:
            layers.append(self._dropblock)

        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._output_channels))
            
        #layers.append(nn.Softmax(dim = 2))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def checkpoint_function(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
    
    def forward(self, x):
            
        if self._use_dropblock:
            self._dropblock.step() # step linear scheduler of dropblock
        
        skip_conns = []
        
        # dummy input to circumvent no gradient problem
        dummy = torch.tensor(1.0, requires_grad=True)
        
        # encoding
        for step in range(self._model_depth):
            x = checkpoint(self.checkpoint_function(self.down_blocks[step][0]), x, dummy) # convolution
            #print(f'Step {step} conv: ', x.size())
            skip_conns.append(x.clone()) # appends a skip connection
            x = checkpoint(self.checkpoint_function(self.down_blocks[step][1]), x, dummy) # pool
            #print(f'Step {step} pool: ', x.size())
        
        # connection
        x = checkpoint(self.checkpoint_function(self.conn_block), x, dummy)
        #print(f'Step conn: ', x.size())
         
        skip_conns = skip_conns[::-1] # reverse list
        for step in range(self._model_depth):
            x = checkpoint(self.checkpoint_function(self.up_blocks[step][0]), x, dummy) # upsample
            #print(f'Step {step} upsample: ', x.size())
            x = self.skip_connection(x, skip_conns[step]) # performs skip connection
            x = checkpoint(self.checkpoint_function(self.up_blocks[step][1]), x, dummy) # conv
            #print(f'Step {step} conv: ', x.size())
            
        x = self.output_conv(x)
        #print(f'Step final: ', x.size())

        # avoid issues
        x = x.clamp(0,1) # clamps output
        x[x!=x] = 0 # avoids nan values
        
        # free up memory
        del skip_conns, dummy
        
        return x

    def save_intermediate_tensors(self, x, save_dir):
            
        skip_conns = []
        
        # dummy input to circumvent no gradient problem
        dummy = torch.tensor(1.0, requires_grad=True)
        
        # encoding
        for step in range(self._model_depth):
            x = checkpoint(self.checkpoint_function(self.down_blocks[step][0]), x, dummy) # convolution
            torch.save(x.detach().clone().cpu(), os.path.join(save_dir, f'convolution{step}.pt'))
            #print(f'Step {step} conv: ', x.size())
            skip_conns.append(x.clone()) # appends a skip connection
            x = checkpoint(self.checkpoint_function(self.down_blocks[step][1]), x, dummy) # pool
            torch.save(x.detach().clone().cpu(), os.path.join(save_dir, f'pool{step}.pt'))
            #print(f'Step {step} pool: ', x.size())
        
        # connection
        x = checkpoint(self.checkpoint_function(self.conn_block), x, dummy)
        torch.save(x.detach().clone().cpu(), os.path.join(save_dir, 'connection.pt'))
        #print(f'Step conn: ', x.size())
         
        skip_conns = skip_conns[::-1] # reverse list
        for step in range(self._model_depth):
            x = checkpoint(self.checkpoint_function(self.up_blocks[step][0]), x, dummy) # upsample
            torch.save(x.detach().clone().cpu(), os.path.join(save_dir, f'upsample{step}.pt'))
            #print(f'Step {step} upsample: ', x.size())
            x = self.skip_connection(x, skip_conns[step]) # performs skip connection
            x = checkpoint(self.checkpoint_function(self.up_blocks[step][1]), x, dummy) # conv
            torch.save(x.detach().clone().cpu(), os.path.join(save_dir, f'deconvolution{step}.pt'))
            #print(f'Step {step} conv: ', x.size())
            
        x = self.output_conv(x)
        torch.save(x.detach().clone().cpu(), os.path.join(save_dir, 'output.pt'))
        #print(f'Step final: ', x.size())

        # avoid issues
        x = x.clamp(0,1) # clamps output
        x[x!=x] = 0 # avoids nan values
        
        # free up memory
        del skip_conns, dummy
        
def set_optimizer(optimizer, network_params, optim_params):
    if optimizer == 'sgd':
        return optim.SGD(network_params, **optim_params)
    elif optimizer == 'rmsprop':
        return optim.RMSprop(network_params, **optim_params)
    elif optimizer == 'adam':
        del optim_params['momentum'] # no momentum
        return optim.Adam(network_params, **optim_params)
    else:
        return optim.SGD(network_params, **optim_params)

def init_mode(network, mode):
    if mode == 'unet':
        network.apply(unet_initialization)
    elif mode == 'kaiming_norm':
        network.apply(init_weight_k_norm)
    elif mode == 'kaiming_uni':
        network.apply(init_weight_k_uni)
    return

def unet_initialization(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight, mean = 0.0, std = (2**(1/3))/9)
        


# This works better with relu or leaky_relu activations (default is leaky_relu, but we use relu)
def init_weight_k_norm(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d: 
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


def init_weight_k_uni(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        
class Sample(nn.Module):
# creates a sample output for the UNET to test structures

    def __init__(self, init_channels, filters, output_channels, model_depth = 4, pool_mode = 'max', up_mode = 'upconv', connection = 'cat',  same_padding = True, use_batchnorm = True, use_dropblock = False, block_size = 0, max_drop_prob = 0, dropblock_ls_steps = 500, conv_layers_per_block = 2, activation_fcn = 'relu', neg_slope = 0):
        """Architecure for a UNET (requires input image dimensions to be factorable by model_depth * 2)

          inputs:
          init_channels (int): number channels original image contains (typically 1 or 3)
          filters (int): number of filters to begin with (all filters for layers following will be multiples of this)
          pool_mode (str): 'max', 'avg', 'conv' are 3 different options for the pooling methodology (max & avg result in upsampling while conv results in ConvTranspose in the decoder)
          up_mode (str): 'upsample', 'upconv' chooses between upsampling or upwards convolution
          connection (str): 'add', 'cat', 'none' are 3 options for handling 
          same_padding (bool): option to keep size same after each conv layer
          use_batchnorm (bool): include batchnorm layers
          conv_layers_per_block (int>1): number of layers per block
          activation_fcn (str): activation function of relu or leaky relu
          neg_slope (float): negative slope of leaky relu (does not affect relu)"""

        super(Sample, self).__init__()

        # check for appropriate inputs
        
        
        # constants
        self._kernel_size = 3 # should we make this variable
        self._stride = 1
        self._pool_kernel = 2
        self._pool_stride = 2
        self._upsample_factor = 2

        # settings
        self.same_padding = same_padding
        self._init_channels = init_channels
        self._filters = filters
        self._output_channels = output_channels
        
        if connection not in ['add', 'cat', 'none' ]: # how the skip connection is done
            print('Connection type must be of (add, cat, none). Defaulting to cat')
            connection = 'cat'
        self._connection = connection

        if same_padding:
            self._padding = 'same'
        else:
            self._padding = 0

        self._use_batchnorm = use_batchnorm
        
        self._use_dropblock = use_dropblock
        if use_dropblock:   
            self._dropblock = LinearScheduler(DropBlock2D(block_size = block_size, drop_prob = 0.),
                                              start_value = 0.,
                                              stop_value = max_drop_prob,
                                              nr_steps = dropblock_ls_steps)
        
        
        if pool_mode not in ['max', 'avg', 'conv']: #pooling mode
            print('Pool Mode must be of (max, avg, conv). Defaulting to max')
            pool_mode = 'max'
        self._pool_mode = pool_mode
        if up_mode not in ['upsample', 'upconv']:
            print('Up_Mode must be of (upsample, upconv). Defaulting to upconv')
            up_mode = 'upconv'
        self._up_mode = up_mode
        
        if conv_layers_per_block <= 1: # number of convolutions in each block
            print('Convolutional Layers in each block must be 2 or more. Defaulting to 2')
            conv_layers_per_block = 2
        self._conv_layers_per_block = conv_layers_per_block
        
        
        
        if activation_fcn == 'relu': # activation functions
            self._act_fcn = nn.ReLU()
        elif activation_fcn == 'leaky_relu':
            self._act_fcn = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            print('Activation Function at each step must be relu or leaky_relu. Defaulting to relu')
            self._act_fcn = nn.ReLU()
        
        self._model_depth = model_depth
        
        
        self._model = nn.Conv2d(3, 2, 3, padding = 1)
        self._out = nn.Sigmoid()
        
        
    def forward(self,x):
        return (self._out(self._model(x)))
    


