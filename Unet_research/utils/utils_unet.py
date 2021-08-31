import torch
from torch import nn 
from torchvision import transforms
from dropblock import DropBlock2D
from icecream import ic


class UNet(nn.Module):
# can convert this to the architecture and move the actual model creation elsewhere

    def __init__(self, init_channels, filters, output_channels, pool_mode = 'max', up_mode = 'upsample', connection = 'cat',  same_padding = True, use_batchnorm = True,  conv_layers_per_block = 2):
        """Architecure for a UNET

          inputs:
          init_channels (int): number channels original image contains (typically 1 or 3)
          filters (int): number of filters to begin with (all filters for layers following will be multiples of this)
          pool_mode (str): 'max', 'avg', 'conv' are 3 different options for the pooling methodology (max & avg result in upsampling while conv results in ConvTranspose in the decoder)
          up_mode (str): 'upsample', 'upconv' chooses between upsampling or upwards convolution
          connection (str): 'add', 'cat', 'none' are 3 options for handling 
          same_padding (bool): option to keep size same after each conv layer
          use_batchnorm (bool): include batchnorm layers
          conv_layers_per_block (int>1): number of layers per block"""

        super(UNet, self).__init__()

        # check for appropriate inputs
        if pool_mode not in ['max', 'avg', 'conv']:
            print('Pool Mode must be of (max, avg, conv)')
        elif connection not in ['add', 'cat', 'none' ]:
            print('Connection type must be of (add, cat, none)')
        elif conv_layers_per_block <= 1:
            print('Convolutional Layers in each block must be 2 or more')

        # constants
        self._kernel_size = 3
        self._stride = 1
        self._pool_kernel = 2
        self._pool_stride = 2
        self._upsample_factor = 2

        # settings
        self.same_padding = same_padding
        self._init_channels = init_channels
        self._filters = filters
        self._output_channels = output_channels
        self._connection = connection

        if same_padding:
            self._padding = 'same'
        else:
            self._padding = 0

        self._use_batchnorm = use_batchnorm
        self._pool_mode = pool_mode
        self._up_mode = up_mode
        self._conv_layers_per_block = conv_layers_per_block


        # actual network
        self.db1_conv, self.db1_pool = self.get_down_block(init=True)
        self.db2_conv, self.db2_pool = self.get_down_block()
        self.db3_conv, self.db3_pool = self.get_down_block()
        self.db4_conv, self.db4_pool = self.get_down_block()
        
        self.conn_block = self.enc_dec_conn_block() # connection from encoder to decoder (dropout?)
        
        self.ub1_upsample, self.ub1_conv = self.get_up_block(out_pad = (1,0))
        self.ub2_upsample, self.ub2_conv = self.get_up_block(out_pad = (0,1))
        self.ub3_upsample, self.ub3_conv = self.get_up_block(out_pad = (0,0))
        self.ub4_upsample, self.ub4_conv = self.get_up_block(out_pad = (0,1))
        
        self.output_conv = self.end_conv()
        

    def enc_dec_conn_block(self):
        layers = []
        
        layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters * 2,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
        
        self._filters *= 2
        
        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters))
            
        layers.append(nn.ReLU())
        
        
        
        # additional convolutional layers in middle of upblock
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
            
            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters))
                
            layers.append(nn.ReLU())

            
        
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
                                  padding=self._padding))
          # no change to filter amount
        else:
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters*2,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
            self._filters = self._filters*2 # change the filter amount

        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters))
            
        layers.append(nn.ReLU())


        # additional convolutional layers to the down block
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))

            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters))
                
            layers.append(nn.ReLU())
        
        
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
                                  stride=self._pool_stride))
            

        if self._use_batchnorm:
            pooling.append(nn.BatchNorm2d(self._filters))
            
        if self._pool_mode == 'conv': # use relu activation before
            pooling.append(nn.ReLU()) # uses relu layers if down sample is a convolution
            
        return nn.Sequential(*layers), nn.Sequential(*pooling)

    
    def get_up_block(self, out_pad = (0,0)):
        '''reverse architecture of downblock'''
        upsample = []
        
        # Upsample Layer
        # Upsample or deconv layer depending on pooling mode
        if self._up_mode == 'upsample': # upsamples, then convolutes once to reduce size
            # upsample
            upsample.append(nn.Upsample(scale_factor=self._upsample_factor))
            upsample.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters//2,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
        elif self._up_mode == 'upconv':
            # conv transpose
            upsample.append(nn.ConvTranspose2d(in_channels=self._filters,
                                           out_channels=self._filters//2,
                                           kernel_size=self._pool_kernel,
                                           stride=self._pool_stride,
                                           output_padding = out_pad))
        
        self._filters //= 2
        
        if self._use_batchnorm:
            upsample.append(nn.BatchNorm2d(self._filters))
            
        upsample.append(nn.ReLU())
        
        
        
        layers = []
        # First convolution after skip connection
        if self._connection in ['none', 'add']:
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
            self._filters //= 2
        elif self._connection == 'cat':
            # concatnation connection (not sure if i built it wrong)
            layers.append(nn.Conv2d(in_channels=self._filters*2,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
        
        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters))
            
        layers.append(nn.ReLU())
        
        # additional convolutional layers in middle of upblock
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))

            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters))
                
            layers.append(nn.ReLU())
            
        return nn.Sequential(*upsample), nn.Sequential(*layers)
        
        
    def skip_connection(self, x, conn):
        ''' outputs x depending on the connection method'''

        crop = transforms.CenterCrop([x.size()[-2], x.size()[-1]]) # crop the image if necessary (if same_padding = False)

        if self._connection == 'cat':
            return torch.cat([x,crop(conn)], dim = 1)
        elif self._connection == 'add':
            return x + crop(conn)
        elif self._connection == 'none':
            return x


    def end_conv(self):
        
        layers = []
        
        layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._output_channels,
                                  kernel_size=1,
                                  stride=self._stride,
                                  padding=self._padding))

        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._output_channels))
            
        #layers.append(nn.Softmax(dim = 2))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        print_mode = True
        # testing - remove later
        if not print_mode:
            ic.disable()
            
        ic('Original:', x.size())
        x = self.db1_conv(x)
        sc1 = x.clone()
        ic('DB1 Conv:', x.size())
        x = self.db1_pool(x)
        ic('DB1 Pool:', x.size())
        x = self.db2_conv(x)
        sc2 = x.clone()
        ic('DB2 Conv:', x.size())
        x = self.db2_pool(x)
        ic('DB2 Pool:', x.size())
        x = self.db3_conv(x)
        sc3 = x.clone()
        ic('DB3 Conv:', x.size())
        x = self.db3_pool(x)
        ic('DB3 Pool:', x.size())
        x = self.db4_conv(x)
        sc4 = x.clone()
        ic('DB4 Conv:', x.size())
        x = self.db4_pool(x)
        ic('DB4 Pool:', x.size())
        
        x = self.conn_block(x)
        ic('Conn Block:', x.size())
        
        x = self.ub1_upsample(x)
        ic('UB1 Upsample:', x.size())
        x = self.skip_connection(x, sc4)
        ic('Skip Conn1:', x.size())
        x = self.ub1_conv(x)
        ic('UB1 Conv:', x.size())
        x = self.ub2_upsample(x)
        ic('UB2 Upsample:', x.size())
        x = self.skip_connection(x, sc3)
        ic('Skip Conn2:', x.size())
        x = self.ub2_conv(x)
        ic('UB2 Conv:', x.size())
        x = self.ub3_upsample(x)
        ic('UB3 Upsample:', x.size())
        x = self.skip_connection(x, sc2)
        ic('Skip Conn3:', x.size())
        x = self.ub3_conv(x)
        ic('UB3 Conv:', x.size())
        x = self.ub4_upsample(x)
        ic('UB4 Upsample:', x.size())
        x = self.skip_connection(x, sc1)
        ic('Skip Conn4:', x.size())
        x = self.ub4_conv(x)
        ic('UB4 Conv:', x.size())
        
        
        x = self.output_conv(x)
        ic('Output:', x.size())
        
        ic.enable()
        return x
    

class UNet2(nn.Module):
# can convert this to the architecture and move the actual model creation elsewhere

    def __init__(self, init_channels, filters, output_channels, model_depth = 4, pool_mode = 'max', up_mode = 'upsample', connection = 'cat',  same_padding = True, use_batchnorm = True, use_dropblock = True, block_size = 7, drop_prob = .1, conv_layers_per_block = 2, activation_fcn = 'relu', neg_slope = .01):
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

        super(UNet2, self).__init__()

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
            self._dropblock = DropBlock2D(block_size = block_size, drop_prob = drop_prob)
        
        
        if pool_mode not in ['max', 'avg', 'conv']: #pooling mode
            print('Pool Mode must be of (max, avg, conv). Defaulting to max')
            pool_mode = 'max'
        self._pool_mode = pool_mode
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
                                  padding=self._padding))
        
        self._filters *= 2
        
        if self._use_dropblock:
            layers.append(self._dropblock)
        
        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters))
            
        layers.append(self._act_fcn)
        
        
        
        # additional convolutional layers in middle of upblock
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
            
            if self._use_dropblock:
                layers.append(self._dropblock)
                
            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters))
                
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
                                  padding=self._padding))
          # no change to filter amount
        else:
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters*2,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
            self._filters = self._filters*2 # change the filter amount

        if self._use_dropblock:
            layers.append(self._dropblock)
            
        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters))
            
        layers.append(self._act_fcn)


        # additional convolutional layers to the down block
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))

            if self._use_dropblock:
                layers.append(self._dropblock)
            
            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters))
                
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
                                  stride=self._pool_stride))
            

        if self._use_batchnorm:
            pooling.append(nn.BatchNorm2d(self._filters))
            
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
                                  padding=self._padding))
        elif self._up_mode == 'upconv':
            # conv transpose
            upsample.append(nn.ConvTranspose2d(in_channels=self._filters,
                                           out_channels=self._filters//2,
                                           kernel_size=self._pool_kernel,
                                           stride=self._pool_stride,))
            
        self._filters //= 2
            
        if self._use_batchnorm:
            upsample.append(nn.BatchNorm2d(self._filters))
            
        upsample.append(self._act_fcn)
        
        
        
        layers = []
        # First convolution after skip connection
        if self._connection in ['none', 'add']:
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
            self._filters //= 2
        elif self._connection == 'cat':
            # concatnation connection (not sure if i built it wrong)
            layers.append(nn.Conv2d(in_channels=self._filters*2,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
        
        if self._use_dropblock:
            layers.append(self._dropblock)
        
        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._filters))
            
        layers.append(self._act_fcn)
        
        # additional convolutional layers in middle of upblock
        for i in range(self._conv_layers_per_block - 1):
            layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._filters,
                                  kernel_size=self._kernel_size,
                                  stride=self._stride,
                                  padding=self._padding))
            
            if self._use_dropblock:
                layers.append(self._dropblock)
            
            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(self._filters))
                
            layers.append(self._act_fcn)
            
        return nn.Sequential(*upsample), nn.Sequential(*layers)
        
        
    def skip_connection(self, x, conn):
        ''' outputs x depending on the connection method'''

        crop = transforms.CenterCrop([x.size()[-2], x.size()[-1]]) # crop the image if necessary (if same_padding = False)

        if self._connection == 'cat':
            return torch.cat([x,crop(conn)], dim = 1)
        elif self._connection == 'add':
            return x + crop(conn)
        elif self._connection == 'none':
            return x


    def end_conv(self):
        
        layers = []
        
        layers.append(nn.Conv2d(in_channels=self._filters,
                                  out_channels=self._output_channels,
                                  kernel_size=1,
                                  stride=self._stride,
                                  padding=self._padding))
        
        if self._use_dropblock:
            layers.append(self._dropblock)

        if self._use_batchnorm:
            layers.append(nn.BatchNorm2d(self._output_channels))
            
        #layers.append(nn.Softmax(dim = 2))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def forward(self, x):
            
        
        skip_conns = []
        
        # encoding
        for step in range(self._model_depth):
            x = self.down_blocks[step][0](x) # convolution
            print(f'Step {step} conv: ', x.size())
            skip_conns.append(x.clone()) # appends a skip connection
            x = self.down_blocks[step][1](x) # pool
            print(f'Step {step} pool: ', x.size())
        
        # connection
        x = self.conn_block(x)
        print(f'Step conn: ', x.size())
         
        skip_conns = skip_conns[::-1] # reverse list
        for step in range(self._model_depth):
            x = self.up_blocks[step][0](x) # upsample
            print(f'Step {step} upsample: ', x.size())
            x = self.skip_connection(x, skip_conns[step]) # performs skip connection
            x = self.up_blocks[step][1](x) # conv
            print(f'Step {step} conv: ', x.size())
            
        x = self.output_conv(x)
        print(f'Step final: ', x.size())
        
        return x


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