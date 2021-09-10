import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import matplotlib.cm as cm
import numpy as np
import os
import torch

from utils.utils_general import TensortoPIL, get_masked, split_target




###### CHANGE THIS LATER

# plotting test epoch
def final_test_metrics(network, val_dataloader, test_dataloader, train_losses, val_losses, device, num_test_samples = 20, save_model = False, use_mask = True, save_path = None):
    
    
    # setup save folders
    model_folder = os.path.join(save_path, 'model')
    loss_folder = os.path.join(save_path, 'losses')
    test_folder = os.path.join(save_path, 'test_images')
    val_folder = os.path.join(save_path, 'val_images')
    
    examples_folder = os.path.join(val_folder , 'examples')
    
    evaluation_folder = os.path.join(val_folder , 'eval')
    
    # create the folders if they don't exist
    if not os.path.exists(loss_folder):
        os.mkdir(loss_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)


    # save model
    if save_model:
        save_model(network=network,
                save_path=model_folder)
        
    # save losses
    save_losses_as_text(train_losses=train_losses,
                        val_losses=val_losses,
                        save_path=loss_folder)
    
    # save loss profile
    save_loss_profile(train_losses=train_losses,
                        val_losses=val_losses,
                        save_path=loss_folder)


    # save model outputs
    network.eval()
    with torch.no_grad():
        
        count = 0
        for test_id, (test_image_batch, _, test_mask) in enumerate(test_dataloader):
            
            # test run
            test_image_batch = test_image_batch.to(device)
            test_segmentation = network(test_image_batch)
            
            # move onto cpu
            test_segmentation = test_segmentation.cpu()
            
            
            # save operations
            for image in range(len(test_image_batch)): # for each image in the batch
                
                # get image id
                im_id = (image+1)+(test_id * len(test_image_batch))
                
                # save an example
                save_example(unbatched_image=test_image_batch[image],
                                unbatched_segmentation=test_segmentation[image],
                                id=im_id,
                                save_path=test_folder
                                )
                count += 1
                # break if > num_test_samples is saved
                if count >= num_test_samples:
                    break
            
            # free up memory
            del test_segmentation, test_image_batch
            
            # break if > num_test_samples is saved
            if count >= num_test_samples:
                break
        
        for val_id, (val_image_batch, val_gt, val_mask) in enumerate(val_dataloader):

            # val run
            val_image_batch = val_image_batch.to(device)
            val_segmentation = network(val_image_batch)
            
            # get gt
            val_gt = split_target(val_gt)
            val_gt = val_gt.to(device)
        
            # if using masks, only get region with mask
            if use_mask:
                val_seg_masked, val_gt_masked, val_mask = get_masked(val_segmentation, val_gt, val_mask, device)
            
            
            # move back onto cpu
            val_seg_masked = val_seg_masked.cpu()
            val_gt_masked = val_gt_masked.cpu()
            val_mask = val_mask.cpu()
            
            # save operations
            for image in range(len(val_image_batch)): # get each image in one image batch
                
                # create a folder
                im_id = (image+1)+(val_id * len(val_image_batch))
                im_folder = os.path.join(val_folder, f"val_image{im_id}")
                if not os.path.exists(im_folder):
                    os.mkdir(im_folder)
                
                # save examples
                save_example(unbatched_image=val_image_batch[image],
                                unbatched_segmentation=val_seg_masked[image],
                                unbatched_gt=val_gt_masked[image],
                                id=im_id,
                                save_path=im_folder
                                )
                
                # save confusion_matrix
                save_confusion_matrix(unbatched_segmentation=val_seg_masked[image],
                                    unbatched_gt=val_gt_masked[image],
                                    unbatched_mask=val_mask[image],
                                    save_path=im_folder)
                
                # save contour map
                save_contour_map(unbatched_segmentation=val_seg_masked[image],
                                 unbatched_gt=val_gt_masked[image],
                                 save_path=im_folder)
    
    
            # free mem
            del val_seg_masked, val_gt_masked, val_mask, val_image_batch, val_gt
    
    



def save_model(network, save_path = '.'):
    torch.save(network.state_dict(), os.path.join(save_path, f'model.pth'))

def save_losses_as_text(train_losses, val_losses, save_path = '.'):
    ''' saves the training and validation losses as a txt file'''
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # saves
    train_losses.tofile(os.path.join(save_path, 'train_losses.txt'), sep = '\n', format = '%ls')
    val_losses.tofile(os.path.join(save_path, 'validation_losses.txt'), sep = '\n', format = '%ls')



def save_loss_profile(train_losses, val_losses, save_path = '.' ):
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
    fig.savefig(os.path.join(save_path, 'loss_profile.png'))
    
    plt.close(fig) # close image for memory issues



def save_contour_map(unbatched_segmentation, unbatched_gt, save_path = '.'):
    ''' creates and saves a contour map between a segmentation class and gt class to observe class mismatches
    unbatched_segmentation and unbatched_gt are a single segmentation and gt of an image batch'''

    def get_diff(segmentation_class, gt_class):
        
        # find distance between segmentation and gt
        segmentation_class = torch.round(segmentation_class) # round to either 0 or 1 for 50% threshold

        # get a diverging difference
        diff = 2 * (segmentation_class - gt_class) / (torch.abs(segmentation_class) + torch.abs(gt_class))
        
        return diff

    
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (25, 10))
    
    # create vessel class
    div1_map = ax1.imshow(get_diff(unbatched_segmentation[0], unbatched_gt[0]), cmap = cm.seismic)
    fig.colorbar(div1_map, ax = ax1)
    ax1.set_title('Divergence Map Vessel Segmentation', fontsize = 12)
    
    # create non-vessel class image
    div2_map = ax2.imshow(get_diff(unbatched_segmentation[1], unbatched_gt[1]), cmap = cm.seismic)
    fig.colorbar(div2_map, ax = ax2)
    ax1.set_title('Divergence Map Non-Vessel Segmentation', fontsize = 12)
    
    
    # save figure
    fig.savefig(os.path.join(save_path, 'contour_map.png'))
        
    plt.close(fig)
    

def save_confusion_matrix(unbatched_segmentation, unbatched_gt, unbatched_mask, save_path = '.'):
    ''' creates and saves a confusion matrix of true/false positives and negatives to observe how many mismatch in classes there are
    inputs should be masked if necessary,
    For pixels within (threshold) of their appropriate class, they are considered true, otherwise if they are within (threshold) from their opposite class, they are false
    all other pixels are classified as undecided'''
    
    def get_true_false(segmentation_class, gt_class, mask):
  
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

        threshold = .5
        # calculate each value
        true_pos = comp.ge(2 - threshold).count_nonzero()
        true_neg = comp.le(0 + threshold).count_nonzero() - ignored_pixels # all masked pixels are true negs
        false_neg = (comp.le(1 + threshold) & comp.ge(1.0)).count_nonzero() # all perfect opposities are false negatives (mild computation error)
        false_pos = (comp.ge(1 - threshold) & comp.lt(1.0)).count_nonzero() 
        return [[true_pos, false_neg], [false_pos, true_neg]]
    

    # set up chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8) )
    
    # get first class conf matrix
    data_vessel = np.array(get_true_false(unbatched_segmentation[0], unbatched_gt[0], unbatched_mask[0]))
    conf_mat1 = ax1.matshow(data_vessel, cmap = cm.coolwarm)
    for (i, j), z in np.ndenumerate(data_vessel):
        ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax1.set_xticklabels([''] + ['True (<.5 off)', f'False (>.5 off'])
    ax1.set_yticklabels([''] + ['True', 'False'])
    ax1.set_title(f"Confusion Matrix (.5 Thresholded) - Vessel Segmentation")
    
    # get Secong class conf matrix
    data_vessel2 = np.array(get_true_false(unbatched_segmentation[1], unbatched_gt[1], unbatched_mask[1]))
    conf_mat2 = ax2.matshow(data_vessel2, cmap = cm.coolwarm)
    for (i, j), z in np.ndenumerate(data_vessel2):
        ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax2.set_xticklabels([''] + ['True (<.5 off)', f'False (>.5 off'])
    ax2.set_yticklabels([''] + ['True', 'False'])
    ax2.set_title(f"Confusion Matrix (.5 Thresholded) - Non-Vessel Segmentation")
    
    # for saving
    fig.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        
    # close figure for memory
    plt.close(fig)
    
    
def save_example(unbatched_image, unbatched_segmentation, unbatched_gt = None, id = 0, save_path = '.'):
    ''' saves the image, segmentation and ground truth of one example (takes in raw tensor form)
        if gt = None, no ground truth will be displayed
        Segmentation and GT should already be in image format (or a list containing images)'''
    
    fig = plt.figure(figsize = (15, 7), tight_layout=True)
    gs = fig.add_gridspec(2, 3)
    
    if unbatched_gt == None:
        # 3 columns
        axes = []
        for i in range(3):
            ax = fig.add_subplot(gs[:, i])
            axes.append(ax)
            
        # add images 
        axes[0].imshow(TensortoPIL(unbatched_image)) # plain image
        axes[0].set_title("Base Image")
        axes[1].imshow(TensortoPIL(unbatched_segmentation[0]), cmap = 'gray') # first class
        axes[1].set_title("Vessel Segmentation")
        axes[2].imshow(TensortoPIL(unbatched_segmentation[1]), cmap = 'gray') # inverse class
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
        axes[0].imshow(TensortoPIL(unbatched_image)) # plain image
        axes[0].set_title("Base Image")
        axes[1].imshow(TensortoPIL(unbatched_segmentation[0]), cmap = 'gray') # first class
        axes[1].set_title("Vessel Segmentation")
        axes[2].imshow(TensortoPIL(unbatched_segmentation[1]), cmap = 'gray') # inverse class
        axes[2].set_title("Non-vessel Segmentation")
        axes[3].imshow(TensortoPIL(unbatched_gt[0]), cmap = 'gray') # first gt
        axes[3].set_title("Vessel GT")
        axes[4].imshow(TensortoPIL(unbatched_gt[1]), cmap = 'gray') # inverse gt
        axes[4].set_title("Non-vessel GT")
    
        fig.suptitle('Validation Example', x = .25)
        name = 'validation'
    

    fig.savefig(os.path.join(save_path, f'{name}_example_{id}.png'))
        
    plt.close(fig)


def plot_network_weights(network, mode = 'avg', save = True, save_path = '.'):
    ''' Plots the weight distribution of each layer in the network to see magnitude of weights
    modes:  avg - takes the avg of each layer's weights/biases
            det - takes a determinant of each layer's weights/biases
            ??? - other metrics
            histogram - overlap 
    '''
    pass