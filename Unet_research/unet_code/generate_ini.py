import sys
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest = 'name', default = 'train_ini' , help = 'Name of our init file')
    filename = parser.parse_args().name
    with open(filename + '.ini', 'w') as fp:
        # model params
        fp.write('[model_params]\n')
        fp.write('init_channels=1\n')
        fp.write('filters=64\n')
        fp.write('output_channels=1\n')
        fp.write('model_depth=4\n')
        fp.write('pool_mode=max\n')
        fp.write('up_mode=upconv\n')
        fp.write('connection=cat\n')
        fp.write('same_padding=True\n')
        fp.write('use_batchnorm=True\n')
        fp.write('use_dropblock=False\n')
        fp.write('conv_layers_per_block=2\n')
        fp.write('activation_fcn=relu\n')
        fp.write('\n')
        fp.write('#These are needed if use_dropblock is True\n')
        fp.write('block_size=0\n')
        fp.write('max_drop_prob=0\n')
        fp.write('dropblock_ls_step=0\n')
        fp.write('#These are needed if activation_fcn is leaky_relu\n')
        fp.write('neg_slope=0\n')
        
        fp.write('\n')
        fp.write('[general_params]\n')
        fp.write('num_epochs=50\n')
        fp.write('train_batch_size=3\n')
        fp.write('val_batch_size=1\n')
        fp.write('test_batch_size=1\n')
        fp.write('initialization=kaiming_norm\n')
        
        fp.write('\n')
        fp.write('[optimizer_params]\n')
        fp.write('optimizer=sgd\n')
        fp.write('lr=.001\n')
        fp.write('#if optimizer is adam, momentum will not be considered\n')
        fp.write('momentum=.99\n')
        
        
        