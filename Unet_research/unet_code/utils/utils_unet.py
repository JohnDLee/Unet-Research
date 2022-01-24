import torch
from torch import nn 
from torchvision import transforms
from torch.nn.functional import pad
import math

from utils.utils_modules import LinearScheduler
from fairscale.nn import checkpoint_wrapper


class UNet(nn.Module):
# can convert this to the architecture and move the actual model creation elsewhere

    def __init__(
        self,
        init_channels: int = 3,
        filters: int = 64, 
        output_channels: int = 1,
        model_depth: int = 4,
        pool_mode: str = 'max',
        up_mode: str = 'upconv',
        connection: str = 'cat',
        same_padding: bool = True,
        conv_layers_per_block: int = 2,
        checkpointing = True
        ):
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

        # settings
        self.same_padding = same_padding
        self._init_channels = init_channels
        self._filters = filters
        self._output_channels = output_channels
        
        if connection not in ['add', 'cat', 'none' ]: # how the skip connection is done
            raise ValueError('Connection type must be of (add, cat, none)')
        self._connection = connection

        if same_padding:
            self._padding = 'same'
        else:
            self._padding = 0

        # set these to identity if we chose not to use.
        self._norm = nn.Identity
        self._bias = True
        self._norm_params = {}
        self._norm_keys = []

        self._dropblock = nn.Identity()
        
        if pool_mode not in ['max', 'avg', 'conv']: #pooling mode
            raise ValueError('Pool Mode must be of (max, avg, conv).')
        self._pool_mode = pool_mode

        if up_mode not in ['upsample', 'upconv']:
            raise ValueError('Up_Mode must be of (upsample, upconv).')
        self._up_mode = up_mode
        
        if conv_layers_per_block <= 1: # number of convolutions in each block
            raise ValueError('Convolutional Layers in each block must be 2 or more.')

        self._conv_layers_per_block = conv_layers_per_block
        
        # default ReLU activation function
        self._act_fcn = nn.ReLU()
        
        self._model_depth = model_depth
        
        self._checkpointing = checkpointing
        
        
        # actual network
    
    def create_model(self,):
        """ Creates the model """
        # add downblocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(nn.ModuleList(self.get_down_block(init=True))) # add first downblock (unique starting condition)
        for i in range(self._model_depth-1):
            self.down_blocks.append(nn.ModuleList(self.get_down_block()))
        
        # add connection blocks
        self.conn_block = self.enc_dec_conn_block() # connection from encoder to decoder (dropout?)
        
        # add upblocks
        self.up_blocks = nn.ModuleList()
        for i in range(self._model_depth):
            self.up_blocks.append(nn.ModuleList(self.get_up_block()))
        
        # add output convolutions
        self.output_conv = self.end_conv()

    def set_dropblock(
        self,
        dropblock_class,
        block_size: int = 7,
        drop_prob: float = .1,
        use_scheduler:bool = True,
        start_drop_prob: float = 0.,
        max_drop_prob:float = .2,
        dropblock_ls_steps: int = 500,
        ):
        """ Sets up dropblock for model """
        if use_scheduler:
            self._dropblock = LinearScheduler(dropblock_class(block_size = block_size, drop_prob = drop_prob),
                                              start_value = start_drop_prob,
                                              stop_value = max_drop_prob,
                                              nr_steps = dropblock_ls_steps)
        else:
            self._dropblock = dropblock_class(block_size = block_size, drop_prob = drop_prob)
    
    def set_normalization(
        self,
        normalization_class,
        params: dict
        ):
        """ sets a normalization layer for model
        if a value in params is "fill", it is filled by self._filters at the moment if class instanciation"""
        self._norm  = normalization_class
        self._bias = False # no bias with norm
        self._norm_params = params
        for key, value in params.items():
            if value == 'fill':
                self._norm_keys.append(key)

    def update_norm_params(self):
        """ replace keys with current channel length"""
        for key in self._norm_keys:
            self._norm_params[key] = self._filters
        
    def set_activation_function(
        self,
        act_fcn
    ):
        """ sets activation function"""
        self._act_fcn = act_fcn

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
        
        # set normalization
        self.update_norm_params()
        layers.append(self._norm(**self._norm_params))
        
        # set dropblock
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
            
            # set normalization
            self.update_norm_params()
            layers.append(self._norm(**self._norm_params))
            
            # set dropblock
            layers.append(self._dropblock)
                
            layers.append(self._act_fcn)

        # if checkpointing is true
        if self._checkpointing:
            return checkpoint_wrapper(nn.Sequential(*layers))
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
            
        # set normalization
        self.update_norm_params()
        layers.append(self._norm(**self._norm_params))
        
        # set dropblock
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

            # set normalization
            self.update_norm_params()
            layers.append(self._norm(**self._norm_params))
            
            # set dropblock
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
            

        # set normalization
        self.update_norm_params()
        pooling.append(self._norm(**self._norm_params))

            
        if self._pool_mode == 'conv': # use relu activation before
            pooling.append(self._act_fcn) # uses relu layers if down sample is a convolution
            
        if self._checkpointing:
            return checkpoint_wrapper(nn.Sequential(*layers)), checkpoint_wrapper(nn.Sequential(*pooling))

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
            
        self.update_norm_params()
        upsample.append(self._norm(**self._norm_params))
            
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
        
        # set normalization
        self.update_norm_params()
        layers.append(self._norm(**self._norm_params))
        
        # set dropblock
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
            
            # set normalization
            self.update_norm_params()
            layers.append(self._norm(**self._norm_params))
            
            # set dropblock
            layers.append(self._dropblock)
                
            layers.append(self._act_fcn)
            
        if self._checkpointing:
            return checkpoint_wrapper(nn.Sequential(*upsample)), checkpoint_wrapper(nn.Sequential(*layers))
        return nn.Sequential(*upsample), nn.Sequential(*layers)
        
        
    def skip_connection(self, x, conn):
        ''' outputs x depending on the connection method'''

        crop = transforms.CenterCrop([x.size()[-2], x.size()[-1]]) # crop the image if necessary (if same_padding = False)

        if self._connection == 'cat':
            x = torch.cat([x,crop(conn)], dim = 1)
            x = self._dropblock(x)
            return x
        elif self._connection == 'add':
            x = x + crop(conn)
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
            
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
            
        if type(self._dropblock) == LinearScheduler and self.training == True:
            self._dropblock.step() # step linear scheduler of dropblock

        x = self.autopad(x) # pad
        skip_conns = []
        
        # encoding
        for step in range(self._model_depth):
            x = self.down_blocks[step][0](x) # convolution
            #print(f'Step {step} conv: ', x.size())
            skip_conns.append(x.clone()) # appends a skip connection
            x = self.down_blocks[step][1](x) # pool
            #print(f'Step {step} pool: ', x.size())
        
        # connection
        x = self.conn_block(x)
        #print(f'Step conn: ', x.size())
         
        skip_conns = skip_conns[::-1] # reverse list
        for step in range(self._model_depth):
            x = self.up_blocks[step][0](x)
            #print(f'Step {step} upsample: ', x.size())
            x = self.skip_connection(x, skip_conns[step]) # performs skip connection
            x = self.up_blocks[step][1](x)# conv
            #print(f'Step {step} conv: ', x.size())
            
        x = self.output_conv(x)
        #print(f'Step final: ', x.size())


        x = self.depad(x)

        # avoid issues
        x = x.clamp(0,1) # clamps output
        x[x!=x] = 0 # avoids nan values
        
        # free up memory
        del skip_conns
        
        return x
    
    def autopad(self, x):
        """Finds nearest multiple of model_depth and pads to that size"""
        multiple = 2 ** self._model_depth
        h, w = x.size()[-2:]
        pad_bottom = math.ceil(h / multiple) * multiple - h
        pad_right = math.ceil(w / multiple) * multiple - w
        self._original_size = (h, w)
        return pad(x, (0, pad_right, 0, pad_bottom))
    
    def depad(self, x):
        """Returns back to original size"""
        h, w = self._original_size
        return x[:,:, :h, :w]