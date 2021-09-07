import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import matplotlib.cm as cm
import numpy as np
import os
import torch

from utils.utils_general import TensortoPIL, get_masked, split_target




###### CHANGE THIS LATER

# plotting test epoch
def final_test_metrics(network, val_dataloader, test_dataloader, train_losses, val_losses, confusion_threshold, device, use_mask = True, save = True, save_path = None):
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # save parameters
        if not os.path.exists(os.path.join(save_path , 'losses')):
            os.mkdir(os.path.join(save_path , 'losses'))
        if not os.path.exists(os.path.join(save_path , 'examples')):
            os.mkdir(os.path.join(save_path , 'examples'))
        if not os.path.exists(os.path.join(save_path , 'model')):
            os.mkdir(os.path.join(save_path, 'model'))
        if not os.path.exists(os.path.join(save_path , 'eval')):
            os.mkdir(os.path.join(save_path, 'eval'))
    save_loss_params = {'save': save, 'save_path': os.path.join(save_path, 'losses')}
    save_example_params = {'save': save, 'save_path': os.path.join(save_path, 'examples')}
    save_model_params = {'save': save, 'save_path': os.path.join(save_path, 'model')}
    save_eval_params = {'save': save, 'save_path': os.path.join(save_path, 'eval',)}

    network.eval()
    with torch.no_grad():
        
        test_id, (test_image_batch, _, test_mask) = next(enumerate(test_dataloader))
        val_id, (val_image_batch, val_gt, val_mask) = next(enumerate(val_dataloader))
        
        # test run
        test_image_batch = test_image_batch.to(device)
        test_segmentation = network(test_image_batch)
        
        

        # val run
        val_image_batch = val_image_batch.to(device)
        val_segmentation = network(val_image_batch)
        
        val_gt = split_target(val_gt)
        val_gt = val_gt.to(device)
        
        
        # if using masks, only get region with mask
        if use_mask:
            val_seg_masked, val_gt_masked, val_mask = get_masked(val_segmentation, val_gt, val_mask, device)
        
    # move back onto cpu
    val_seg_masked = val_seg_masked.cpu()
    val_gt_masked = val_gt_masked.cpu()
    val_mask = val_mask.cpu()
    test_segmentation = test_segmentation.cpu()
    
    # save model
    save_model(network=network,
                **save_model_params)
    
    # save losses
    save_losses_as_text(train_losses=train_losses,
                        val_losses=val_losses,
                        **save_loss_params)
    
    # save loss profile
    save_loss_profile(train_losses=train_losses,
                        val_losses=val_losses,
                        **save_loss_params)
    
    # make a contour map for class 1 and class 2
    save_contour_map(segmentation_class=val_seg_masked[0][0],
                    gt_class=val_gt_masked[0][0],
                    class_id='vessel',
                    **save_eval_params)
                    
    save_contour_map(segmentation_class=val_seg_masked[0][1],
                        gt_class=val_gt_masked[0][1],
                        class_id='non-vessel',
                        **save_eval_params)
    
    # save confusion matrix
    threshold = confusion_threshold
    save_confusion_matrix(segmentation_class=val_seg_masked[0][0],
                        gt_class=val_gt_masked[0][0],
                        mask=val_mask[0][0],
                        class_id='vessel',
                        threshold=threshold,
                        **save_eval_params)
    
    save_confusion_matrix(segmentation_class=val_seg_masked[0][1],
                        gt_class=val_gt_masked[0][1],
                        mask=val_mask[0][1],
                        class_id='non-vessel',
                        threshold=threshold,
                        **save_eval_params)
    
    # save test examples
    save_example(image=val_image_batch,
                    segmentation=val_seg_masked,
                    gt=val_gt,
                    **save_example_params
                    )
    save_example(image=test_image_batch,
                    segmentation=test_segmentation,
                    **save_example_params
                    )







def save_model(network, save = True, save_path = '.'):
    if save:
        torch.save(network.state_dict(), os.path.join(save_path, f'model.pth'))
    else:
        print(network.state_dict())

def save_losses_as_text(train_losses, val_losses, save = True, save_path = '.'):
    ''' saves the training and validation losses as a txt file'''
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # saves or displays
    if save:
        train_losses.tofile(os.path.join(save_path, 'train_losses.txt'), sep = '\n', format = '%ls')
        val_losses.tofile(os.path.join(save_path, 'validation_losses.txt'), sep = '\n', format = '%ls')
    else:
        print('Train Losses:', train_losses)
        print('Val Losses:', val_losses)


def save_loss_profile(train_losses, val_losses, save = True, save_path = '.' ):
    ''' saves a graph of the loss profile'''
    
    fig, ax= plt.subplots(1, 1, figsize=(8, 5))
    
    # create plot of validation and training losses
    ax.plot(train_losses, 'b', label = 'Train Losses')
    ax.plot(val_losses, 'r^', label = 'Validation Losses')
    fig.legend(loc = 'upper right')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCELoss')
    fig.suptitle('Loss Profile')
    
    # save or display (save will display too)
    if save:
        fig.savefig(os.path.join(save_path, 'loss_profile.png'))
    else:    
        fig.show()
    
    plt.close(fig) # close image for memory issues
        
        
def save_contour_map(segmentation_class, gt_class, class_id, save = True, save_path = '.'):
    ''' creates and saves a contour map between a segmentation class and gt class to observe class mismatches
    segmentation_class and gt_class are a single corresponding class between the segmentation and gt'''
    '''
    fig, ax = plt.subplots(1, 2)
    a = ax[0].imshow(segmentation_class, cmap = "gray")
    ax[0].set_title("Segmentaiton")
    b = ax[1].imshow(gt_class, cmap = "gray")
    ax[1].set_title("GT")
    fig.colorbar(a, ax = ax[0])
    fig.colorbar(b, ax = ax[1])
    fig.show()
    
    fig , ax = plt.subplots(1, 3 )
    '''
    # find distance between segmentation and gt
    segmentation_class = torch.round(segmentation_class) # round to either 0 or 1 for 50% threshold
    

    diff = 2 * (segmentation_class - gt_class) / (torch.abs(segmentation_class) + torch.abs(gt_class))
    '''
    diff2 = segmentation_class - gt_class
    a = ax[0].imshow(diff2, cmap = cm.seismic, vmin = -1, vmax = 1)
    b = ax[1].imshow(segmentation_class, cmap = "gray")
    ax[1].set_title("Segmentaiton")
    c = ax[2].imshow(gt_class, cmap = "gray")
    ax[2].set_title("GT")
    fig.colorbar(b, ax = ax[1])
    fig.colorbar(c, ax = ax[2])
    fig.colorbar(a, ax = ax[0])
    ax[0].set_title("Diff")
    fig.show()
    '''
    
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    div_map = ax.imshow(diff, cmap = cm.seismic)
    fig.colorbar(div_map, ax = ax)
    fig.suptitle(f'Divergence Map Class {class_id} ', fontsize = 14)
    
    if save:
        fig.savefig(os.path.join(save_path, f'contour_map_class_{class_id}.png'))
    else:
        fig.show()
        
    plt.close(fig)

def save_confusion_matrix(segmentation_class, gt_class, mask, class_id, threshold = .25, save = True, save_path = '.'):
    ''' creates and saves a confusion matrix of true/false positives and negatives to observe how many mismatch in classes there are
    inputs should be masked if necessary,
    For pixels within (threshold) of their appropriate class, they are considered true, otherwise if they are within (threshold) from their opposite class, they are false
    all other pixels are classified as undecided'''
    
  
    comp = segmentation_class + gt_class
    # 2 (- threshold) if true pos
    # 0 (+ threshold) if true neg
    # 1 + (threshold) > x > 1 if false neg
    # 1 > x > 1 - (threshold) if false pos 
    # between true pos/false neg is undecided pos
    # between true neg/false pos is undecided neg
    
    # for masking
    ignored_pixels = 0
    if mask != None:
        ignored_pixels = mask.numel() - mask.count_nonzero()

    # calculate each value
    true_pos = comp.ge(2 - threshold).count_nonzero()
    true_neg = comp.le(0 + threshold).count_nonzero() - ignored_pixels # all masked pixels are true negs
    false_neg = (comp.le(1 + threshold) & comp.ge(1.0)).count_nonzero() # all perfect opposities are false negatives (mild computation error)
    false_pos = (comp.ge(1 - threshold) & comp.lt(1.0)).count_nonzero() 
    undecided_neg = (comp.gt(0 + threshold) & comp.lt(1 - threshold)).count_nonzero() 
    undecided_pos = (comp.lt(2 - threshold) & comp.gt(1 + threshold)).count_nonzero()
    

    
    # set up chart
    data = np.array([[true_pos, false_neg, undecided_pos],[false_pos, true_neg, undecided_neg]])
    fig, ax = plt.subplots(1, 1, figsize = (8, 8) )
    
    conf_mat = ax.matshow(data, cmap = cm.coolwarm)
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax.set_xticklabels([''] + [f'True (<.{threshold} off)', f'False (>.{1-threshold} off', 'Undecided(In between)'])
    ax.set_yticklabels([''] + ['True', 'False'])
    
    fig.suptitle(f"Confusion Matrix - {class_id}")
    
    # for saving
    if save:
        if not os.path.exists(os.path.join(save_path, f'conf_matrix_{class_id}')):
            os.mkdir(os.path.join(save_path, f'conf_matrix_{class_id}'))
        fig.savefig(os.path.join(save_path, f'conf_matrix_{class_id}/conf_matrix.png'))
        with open(os.path.join(save_path, f'conf_matrix_{class_id}/conf_maxtrix_stats.txt'), 'w') as f:
            f.write(f'Accuracy [(TP+TN) / Total]  = {(true_pos + true_neg) / mask.count_nonzero()} \n')
            f.write(f'Error Rate [(FP+FN+UP+UN) / Total] = {(false_pos + false_neg  + undecided_pos + undecided_neg) / mask.count_nonzero()} \n')
            f.write(f'Precision [TP / Total Yeses] = {true_pos / (true_pos + false_pos + undecided_pos)} \n')
            # can add some more...
    else:
        fig.show()
    
    plt.close(fig)
    
    
def save_example(image, segmentation, gt = None, save = True, save_path = '.'):
    ''' saves the image, segmentation and ground truth of one example (takes in raw tensor form)
        if gt = None, no ground truth will be displayed
        Segmentation and GT should already be in image format (or a list containing images)'''
    
    fig = plt.figure(figsize = (15, 7), tight_layout=True)
    gs = fig.add_gridspec(2, 3)
    
    if gt == None:
        # 3 columns
        axes = []
        for i in range(3):
            ax = fig.add_subplot(gs[:, i])
            axes.append(ax)
            
        # add images 
        axes[0].imshow(TensortoPIL(image[0])) # plain image
        axes[0].set_title("Base Image")
        axes[1].imshow(TensortoPIL(segmentation[0][0]), cmap = 'gray') # first class
        axes[1].set_title("Vessel Segmentation")
        axes[2].imshow(TensortoPIL(segmentation[0][1]), cmap = 'gray') # inverse class
        axes[2].set_title("Non-vessel Segmentation")
        
        fig.suptitle('Test Example')
        name = 'test'
    else:
        # 3 columns but middle and right have two rows
        axes = []
        axes.append(fig.add_subplot(gs[:, 0]))
        for i in range(2):
            for j in range(1, 3):
                ax = fig.add_subplot(gs[i,j])
                axes.append(ax)

        # add images 
        axes[0].imshow(TensortoPIL(image[0])) # plain image
        axes[0].set_title("Base Image")
        axes[1].imshow(TensortoPIL(segmentation[0][0]), cmap = 'gray') # first class
        axes[1].set_title("Vessel Segmentation")
        axes[2].imshow(TensortoPIL(segmentation[0][1]), cmap = 'gray') # inverse class
        axes[2].set_title("Non-vessel Segmentation")
        axes[3].imshow(TensortoPIL(gt[0][0]), cmap = 'gray') # first gt
        axes[3].set_title("Vessel GT")
        axes[4].imshow(TensortoPIL(gt[0][1]), cmap = 'gray') # inverse gt
        axes[4].set_title("Non-vessel GT")
    
        fig.suptitle('Validation Example', x = .25)
        name = 'validation'
    
    if save:
        fig.savefig(os.path.join(save_path, f'{name}_example.png'))
    else:
        fig.show()
        
    plt.close(fig)


def plot_network_weights(network, mode = 'avg', save = True, save_path = '.'):
    ''' Plots the weight distribution of each layer in the network to see magnitude of weights
    modes:  avg - takes the avg of each layer's weights/biases
            det - takes a determinant of each layer's weights/biases
            ??? - other metrics
            histogram - overlap 
    '''
    pass