import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import os
from os.path import join, exists
from PIL import Image
import torch
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from zipfile import ZipFile
from utils.utils_general import toPIL
import shutil

def final_test_metrics(trainer,model, val_dataloader, test_dataloader, save_path = None, disable_test = False):
    
    
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

    train_losses=[]
    if 'train_loss_epoch' in trainer.logged_metrics:
        train_losses = trainer.logged_metrics["train_loss_epoch"]
    val_losses=[]
    if 'val_loss_epoch' in trainer.logged_metrics:
        val_losses = trainer.logged_metrics["val_loss_epoch"]

    # save losses
    
    save_losses_as_text(train_losses=train_losses,
                        val_losses=val_losses,
                        save_path=loss_folder)
    
    # save loss profile
    save_loss_profile(train_losses=train_losses,
                        val_losses=val_losses,
                        save_path=loss_folder)
    print("Saved Losses")
    
    if disable_test == False:
        # test data to just 1 image per batch
        test_dataloader = DataLoader(test_dataloader.dataset, batch_size = 1, shuffle = False)
        # save test outputs
        test_data = trainer.predict(model, dataloaders = [test_dataloader])

        # segmentation images folder
        test_segmentations = join(test_folder, 'segmentations')
        if not exists(test_segmentations):
            os.mkdir(test_segmentations)
        test_examples = join(test_folder, 'examples')
        if not exists(test_examples):
            os.mkdir(test_examples)

        for im_id, seg, im, _, in test_data:
            
            # unbatch seg. & image
            im = im[0]
            seg = seg[0]
            im_id += 1
            # save a comparison
            save_test_example(image=im,
                            segmentation=seg,
                            id=im_id,
                            save_path=test_examples,
                            )
            # save image alone
            save_segmentation(segmentation=seg, 
                                id = im_id,
                                save_path=test_segmentations)
        
        print("Saved Test Data")
        # preserve memory
        del test_data

    # val data to just 1 image per batch
    val_dataloader = DataLoader(val_dataloader.dataset, batch_size = 1, shuffle = False)
    # save val outputs
    val_data = trainer.predict(model, dataloaders = [val_dataloader])

    # segmentation images folder
    val_examples = join(val_folder, 'examples')
    if not exists(val_examples):
        os.mkdir(val_examples)

    scores_dict = {'Validation_Image':[], 'F1_Vessel':[], 'AUROC_Vessel':[], 'Accuracy_Vessel':[]}
    for im_id, seg, im, gt in val_data:

        im = im[0]
        seg = seg[0]
        gt = gt[0]
        im_id += 1

        im_folder = join(val_examples, f"val_image_{im_id}")
        if not exists(im_folder):
            os.mkdir(im_folder)
                
                
        # save examples
        save_val_example(image=im,
                        segmentation=seg,
                        gt = gt,
                        id=im_id,
                        save_path=im_folder,
                        )
        # save contour map
        save_contour_map(segmentation=seg,
                        gt=gt,
                        save_path=im_folder
                        )
        # save overlap map
        save_overlap_map(segmentation=seg,
                        gt=gt,
                        save_path=im_folder)

    
    
        # save AUCROC F1 DICE to df
        f1, auroc, accu = get_accuracy_metrics(segmentation=seg,
                                    gt = gt)
        scores_dict['Validation_Image'].append(im_id)
        scores_dict['F1_Vessel'].append(f1)
        scores_dict['AUROC_Vessel'].append(auroc)
        scores_dict['Accuracy_Vessel'].append(accu)
    print("Saved Val Data")
    # save our DF
    scores_df = pd.DataFrame(scores_dict)
    scores_df.to_csv(os.path.join(val_folder, 'metrics.csv'), index = False)
        
    print("Saved All Metrics")
    



    

def get_accuracy_metrics(segmentation, gt):
    ''' returns (f1 score vessel,  auroc , accuracy 0) '''
    
    # threshold
    rounded_seg = torch.round(segmentation).flatten().numpy()
    long_gt = gt.long().flatten().numpy()
    
    # f1 score
    f1 = metrics.f1_score(y_true = long_gt, y_pred = rounded_seg)
    # auroc
    auroc = metrics.roc_auc_score(y_true = long_gt, y_score = rounded_seg)
    # accuracy
    accu = metrics.accuracy_score(y_true = long_gt, y_pred = rounded_seg,)
    return f1, auroc, accu

    
def save_losses_as_text(train_losses, val_losses, save_path = '.'):
    ''' saves the training and validation losses as a txt file'''
    print(train_losses)
    print(val_losses)
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



def save_contour_map(segmentation, gt, save_path = '.'):
    ''' creates and saves a contour map between a segmentation and gt to observe class mismatches'''

    def get_diff(segmentation_class, gt_class):
        # find distance between segmentation and gt
        segmentation_class = torch.round(segmentation_class) # round to either 0 or 1 for 50% threshold
        # get a diverging difference
        diff = 2 * (segmentation_class - gt_class) / torch.clamp((torch.abs(segmentation_class) + torch.abs(gt_class)), min = 0.000001)
        return diff

    
    fig, (ax1) = plt.subplots(1, 1, figsize = (10, 10))
    
    # create vessel class
    diff1 = get_diff(segmentation[0], gt[0])
    div1_map = ax1.imshow(diff1, cmap = cm.seismic)
    fig.colorbar(div1_map, ax = ax1)
    ax1.set_title('Divergence Map Vessel Segmentation', fontsize = 12)
    
    # save figure
    fig.savefig(join(save_path, 'contour_map.png'))
        
    plt.close(fig)
    
    
def save_overlap_map(segmentation, gt, save_path = '.'):
    
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    
    # create numpy arrays of thresholded value
    mask = torch.round(segmentation[0]).numpy()
    # create mask
    masked = np.ma.masked_where(mask == 0, mask)
    
    # red CLIST
    cdict = {'red': ((0, 1, 1),
                   (1, 1, 1)),
           'green': ((0, 0, 0),
                    (1, 0, 0)),
           'blue': ((0, 0, 0),
                   (1, 0, 0))}
    
    # create vessel class
    ax.imshow(toPIL(gt), cmap = 'gray')
    ax.imshow(masked, cmap = LinearSegmentedColormap('custom_cmap', cdict), alpha = .9)
    ax.set_title('Overlap Vessel Segmentation', fontsize = 12)
    # save figure
    fig.savefig(join(save_path, 'overlap_map.png'))
    plt.close(fig)


#patched

def save_test_example(image, segmentation, id, save_path):
    ''' saves test segmentation side by side with original'''

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6), tight_layout=True)
    
    # base image
    axes[0].imshow(toPIL(image), cmap = 'gray') # plain image
    axes[0].set_title("Base Image")
    axes[1].imshow(toPIL(segmentation), cmap = 'gray',) # first class
    axes[1].set_title("Vessel Segmentation")
    fig.suptitle(f'Test Image {id}')
    
    fig.savefig(os.path.join(save_path, f'test_example_{id}.png'))
    plt.close(fig)

def save_segmentation(segmentation, id, save_path):
    """save segmentation alone as a binary png"""
    toPIL(torch.round(segmentation)).convert('L').save(join(save_path, f'{id}.png'))


def save_val_example(image, segmentation, gt, id, save_path):
    ''' saves the image, segmentation and ground truth of one example (takes in raw tensor form)
        if gt = None, no ground truth will be displayed
        Segmentation and GT should already be in image format (or a list containing images)'''

    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 9), tight_layout=True)
    
    # base image
    axes[0].imshow(toPIL(image), cmap = 'gray') # plain image
    axes[0].set_title("Base Image")
    axes[1].imshow(toPIL(segmentation), cmap = 'gray') # first class
    axes[1].set_title("Vessel Segmentation")
    axes[2].imshow(toPIL(gt), cmap = 'gray') # first class
    axes[2].set_title("Vessel Ground Truth")
    fig.suptitle(f'Val Image {id}')
    
    fig.savefig(os.path.join(save_path, f'val_example_{id}.png'))
    plt.close(fig)
