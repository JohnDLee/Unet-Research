import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import os
import torch
import optuna
import optuna.visualization as ov
import sklearn.datasets
import sklearn.metrics
import plotly
import kaleido
import joblib
import sklearn.metrics as metrics

from utils.utils_general import TensortoPIL, get_masked, split_target
from utils.utils_dataset import dePad





    




###### CHANGE THIS LATER

# plotting test epoch
def final_test_metrics(network, val_dataloader, test_dataloader, train_losses, val_losses, device, num_test_samples = 20, use_mask = True, save_path = None):
    
    
    # setup save folders
    loss_folder = os.path.join(save_path, 'losses')
    test_folder = os.path.join(save_path, 'test_images')
    val_folder = os.path.join(save_path, 'val_images')

    
    # create the folders if they don't exist
    if not os.path.exists(loss_folder):
        os.mkdir(loss_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)



    
    # save losses
    save_losses_as_text(train_losses=train_losses,
                        val_losses=val_losses,
                        save_path=loss_folder)
    
    # save loss profile
    save_loss_profile(train_losses=train_losses,
                        val_losses=val_losses,
                        save_path=loss_folder)
    
    # load the model at best point
    network.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
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
        
        
        fig_images = {'original':[],
                      'vessel_seg':[],
                      'vessel_gt':[],
                      'contour_map_vessel':[],
                      'overlap_map_seg':[],
                      'overlap_map_gt':[],
                      'non_vessel_seg':[],
                      'non_vessel_gt':[],
                      'contour_map_non_vessel':[],
                      }
        scores_dict = {'Validation_Image':[], 'F1_Vessel':[], 'AUROC_Vessel':[], 'Accuracy_Vessel':[]}
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
                images = save_example(unbatched_image=val_image_batch[image],
                                    unbatched_segmentation=val_seg_masked[image],
                                    unbatched_gt=val_gt_masked[image],
                                    id=im_id,
                                    save_path=im_folder
                                    )
                fig_images['original'].append(images[0])
                fig_images['vessel_seg'].append(images[1])
                fig_images['non_vessel_seg'].append(images[2])
                fig_images['vessel_gt'].append(images[3])
                fig_images['non_vessel_gt'].append(images[4])

                # save confusion_matrix
                save_confusion_matrix(unbatched_segmentation=val_seg_masked[image],
                                    unbatched_gt=val_gt_masked[image],
                                    unbatched_mask=val_mask[image],
                                    save_path=im_folder)
                
                # save contour map
                contours = save_contour_map(unbatched_segmentation=val_seg_masked[image],
                                 unbatched_gt=val_gt_masked[image],
                                 save_path=im_folder)
                fig_images['contour_map_vessel'].append(contours[0])
                fig_images['contour_map_non_vessel'].append(contours[1])
                
                # save overlap map
                save_overlap_map(unbatched_segmentation=val_seg_masked[image],
                                 unbatched_gt=val_gt_masked[image],
                                 save_path=im_folder)
                fig_images['overlap_map_seg'].append(val_seg_masked[image])
                fig_images['overlap_map_gt'].append(val_gt_masked[image])
    
    
                # save AUCROC F1 DICE to df
                scores = get_accuracy_metrics(unbatched_segmentation=val_seg_masked[image],
                                            unbatched_gt = val_gt_masked[image],
                                            unbatched_mask = val_mask[image])
                scores_dict['Validation_Image'].append(im_id)
                scores_dict['F1_Vessel'].append(scores[0])
                scores_dict['AUROC_Vessel'].append(scores[1])
                scores_dict['Accuracy_Vessel'].append(scores[2])
                
    
    
            # free mem
            del val_seg_masked, val_gt_masked, val_mask, val_image_batch, val_gt
    
        # make/save a figure containing all the things from 
        rows = len(fig_images['original'])
        cols = len(fig_images)
        fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (cols * 7, rows * 7))
        for row in range(rows):
            axes[row][0].imshow(fig_images['original'][row])
            axes[row][1].imshow(fig_images['vessel_seg'][row], cmap = 'gray')
            axes[row][2].imshow(fig_images['vessel_gt'][row], cmap = 'gray')
            axes[row][3].imshow(fig_images['contour_map_vessel'][row], cmap = cm.seismic)
            axes[row][4].imshow(TensortoPIL(dePad(fig_images['overlap_map_gt'][row][0])), cmap = 'gray')
            axes[row][4].imshow(TensortoPIL(dePad(fig_images['overlap_map_seg'][row][0])), cmap = 'Reds', alpha = .5)
            axes[row][5].imshow(fig_images['non_vessel_seg'][row], cmap = 'gray')
            axes[row][6].imshow(fig_images['non_vessel_gt'][row], cmap = 'gray')
            axes[row][7].imshow(fig_images['contour_map_non_vessel'][row], cmap = cm.seismic)
            axes[row][8].imshow(TensortoPIL(dePad(fig_images['overlap_map_gt'][row][1])), cmap = 'gray')
            axes[row][8].imshow(TensortoPIL(dePad(fig_images['overlap_map_seg'][row][1])), cmap = 'Reds', alpha = .5)
        axes[0][0].set_title("Original")
        axes[0][1].set_title("Vessel Seg")
        axes[0][2].set_title("Vessel GT")
        axes[0][3].set_title("Contour Map-Vessel")
        axes[0][4].set_title("Overlap Map-Vessel")
        axes[0][5].set_title("Nonvessel Seg")
        axes[0][6].set_title("Nonvessel GT")
        axes[0][7].set_title("Contour Map-Nonvessel")
        axes[0][8].set_title("Overlap Map-Nonvessel")
    
        # save figure
        fig.savefig(os.path.join(val_folder, 'all_validations.png'))
        plt.close(fig)
        
        # save our DF
        scores_df = pd.DataFrame(scores_dict)
        scores_df.to_csv(os.path.join(val_folder, 'scores.csv'), index = False)
        
        
    
    return


def save_optimization_metrics(study, save_path = '.'):
    ''' Saving Optimization Metrics '''
    optim_path = os.path.join(save_path, 'optimizations')
    if not os.path.exists(optim_path):
        os.mkdir(optim_path)
        
    ov.plot_optimization_history(study,).write_image(os.path.join(optim_path, 'optimization_history.png'))
    ov.plot_intermediate_values(study).write_image(os.path.join(optim_path, 'intermediate_values.png'))
    ov.plot_parallel_coordinate(study, params=['lr', 'model_depth', 'optimizer', 'train_batch_size']).write_image(os.path.join(optim_path, 'parallel_coordinate.png'))
    ov.plot_contour(study, params=['lr', 'model_depth', 'optimizer', 'train_batch_size']).write_image(os.path.join(optim_path, 'contours.png'))
    ov.plot_slice(study).write_image(os.path.join(optim_path, 'slices.png'))
    ov.plot_param_importances(study).write_image(os.path.join(optim_path, 'hyperparam_importances.png'))
    ov.plot_edf(study).write_image(os.path.join(optim_path, 'edf.png'))
    


def get_accuracy_metrics(unbatched_segmentation, unbatched_gt, unbatched_mask):
    ''' returns (f1 score vessel,  auroc , accuracy 0) '''
    
    temp_mask = unbatched_mask[0].long().flatten().numpy()
    temp_seg = torch.round(unbatched_segmentation[0]).flatten().numpy()
    temp_gt = unbatched_gt[0].long().flatten().numpy()
    
    '''
    rounded_seg = []
    long_gt = []
    # CHECK FOR ONLY MASK IMAGE.
    for pixel_index in range(len(temp_mask)):
        if temp_mask[pixel_index] > 0.5: # should be 1 anyways
            rounded_seg.append(temp_seg[pixel_index])
            long_gt.append(temp_gt[pixel_index])
    '''
    
    rounded_seg = temp_seg#np.array(rounded_seg)
    long_gt = temp_gt #np.array(long_gt)
    
    scores = []
    # f1 score
    f1 = metrics.f1_score(y_true = long_gt, y_pred = rounded_seg)
    scores.append(f1)
    
    # auroc
    auroc = metrics.roc_auc_score(y_true = long_gt, y_score = rounded_seg)
    scores.append(auroc)
    
    # accuracy
    accu = metrics.accuracy_score(y_true = long_gt, y_pred = rounded_seg,)
    scores.append(accu)
    
    del rounded_seg, long_gt
    
    return scores
    
    
    
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
    diff1 = get_diff(dePad(unbatched_segmentation[0]), dePad(unbatched_gt[0]))
    div1_map = ax1.imshow(diff1, cmap = cm.seismic)
    fig.colorbar(div1_map, ax = ax1)
    ax1.set_title('Divergence Map Vessel Segmentation', fontsize = 12)
    
    # create non-vessel class image
    diff2 = get_diff(dePad(unbatched_segmentation[1]), dePad(unbatched_gt[1]))
    div2_map = ax2.imshow(diff2, cmap = cm.seismic)
    fig.colorbar(div2_map, ax = ax2)
    ax1.set_title('Divergence Map Non-Vessel Segmentation', fontsize = 12)
    
    
    # save figure
    fig.savefig(os.path.join(save_path, 'contour_map.png'))
        
    plt.close(fig)
    
    return diff1, diff2
    
def save_overlap_map(unbatched_segmentation, unbatched_gt, save_path = '.'):
    
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (25, 10))
    
    # create numpy arrays of thresholded value
    mask0 = torch.round(dePad(unbatched_segmentation[0])).numpy()
    mask1 = torch.round(dePad(unbatched_segmentation[1])).numpy()
    
    # create mask
    masked0 = np.ma.masked_where(mask0 == 0, mask0)
    masked1 = np.ma.masked_where(mask1 == 0, mask1)
    
    # red CLIST
    cdict = {'red': ((0, 1, 1),
                   (1, 1, 1)),
           'green': ((0, 0, 0),
                    (1, 0, 0)),
           'blue': ((0, 0, 0),
                   (1, 0, 0))}
    
    # create vessel class
    ax1.imshow(TensortoPIL(dePad(unbatched_gt[0])), cmap = 'gray')
    ax1.imshow(masked0, cmap = LinearSegmentedColormap('custom_cmap', cdict), )#alpha = .5)
    ax1.set_title('Overlap Vessel Segmentation', fontsize = 12)
    
    # create non-vessel class image
    ax2.imshow(TensortoPIL(dePad(unbatched_gt[1])), cmap = 'gray')
    ax2.imshow(masked1, cmap = LinearSegmentedColormap('custom_cmap', cdict), )#alpha = .5)
    ax2.set_title('Overlap Non-Vessel Segmentation', fontsize = 12)
    
    # save figure
    fig.savefig(os.path.join(save_path, 'overlap_map.png'))
        
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
    data_vessel = np.array(get_true_false(dePad(unbatched_segmentation[0]), dePad(unbatched_gt[0]), dePad(unbatched_mask[0])))
    conf_mat1 = ax1.matshow(data_vessel, cmap = cm.coolwarm)
    for (i, j), z in np.ndenumerate(data_vessel):
        ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax1.set_xticklabels([''] + ['True (<.5 off)', f'False (>.5 off'])
    ax1.set_yticklabels([''] + ['True', 'False'])
    ax1.set_title(f"Confusion Matrix (.5 Thresholded) - Vessel Segmentation")
    
    # get Secong class conf matrix
    data_vessel2 = np.array(get_true_false(dePad(unbatched_segmentation[1]), dePad(unbatched_gt[1]), dePad(unbatched_mask[1])))
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
    
    tuple_images = []
    fig = plt.figure(figsize = (15, 7), tight_layout=True)
    gs = fig.add_gridspec(2, 3)
    
    if unbatched_gt == None:
        # 3 columns
        axes = []
        for i in range(3):
            ax = fig.add_subplot(gs[:, i])
            axes.append(ax)
            
        # add images 
        tuple_images.append(TensortoPIL(dePad(unbatched_image)))
        axes[0].imshow(tuple_images[0]) # plain image
        axes[0].set_title("Base Image")
        tuple_images.append(TensortoPIL(dePad(unbatched_segmentation[0])))
        axes[1].imshow(tuple_images[1], cmap = 'gray') # first class
        axes[1].set_title("Vessel Segmentation")
        tuple_images.append(TensortoPIL(dePad(unbatched_segmentation[1])))
        axes[2].imshow(tuple_images[2], cmap = 'gray') # inverse class
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
        tuple_images.append(TensortoPIL(dePad(unbatched_image)))
        axes[0].imshow(tuple_images[0]) # plain image
        axes[0].set_title("Base Image")
        tuple_images.append(TensortoPIL(dePad(unbatched_segmentation[0])))
        axes[1].imshow(tuple_images[1], cmap = 'gray') # first class
        axes[1].set_title("Vessel Segmentation")
        tuple_images.append(TensortoPIL(dePad(unbatched_segmentation[1])))
        axes[2].imshow(tuple_images[2], cmap = 'gray') # inverse class
        axes[2].set_title("Non-vessel Segmentation")
        tuple_images.append(TensortoPIL(dePad(unbatched_gt[0])))
        axes[3].imshow(tuple_images[3], cmap = 'gray') # first gt
        axes[3].set_title("Vessel GT")
        tuple_images.append(TensortoPIL(dePad(unbatched_gt[1])))
        axes[4].imshow(tuple_images[4], cmap = 'gray') # inverse gt
        axes[4].set_title("Non-vessel GT")
    
        fig.suptitle('Validation Example', x = .25)
        name = 'validation'
    

    fig.savefig(os.path.join(save_path, f'{name}_example_{id}.png'))
        
    plt.close(fig)
    
    return tuple_images
    


def plot_network_weights(network, mode = 'avg', save = True, save_path = '.'):
    ''' Plots the weight distribution of each layer in the network to see magnitude of weights
    modes:  avg - takes the avg of each layer's weights/biases
            det - takes a determinant of each layer's weights/biases
            ??? - other metrics
            histogram - overlap 
    '''
    pass