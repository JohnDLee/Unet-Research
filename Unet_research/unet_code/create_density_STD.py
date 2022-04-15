# %%
import sys
import os
from os.path import join
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ipywidgets import interact_manual,interact, fixed
import math
import cv2
import pandas as pd
import torchvision.transforms as transforms
from sklearn.neighbors import KernelDensity
import torchvision.transforms.functional as TF

# %% [markdown]
# # Functions

# %%
def compare_tensor(tensor_dict1:dict, tensor_dict2:dict, im_dict:dict, target_dict:dict, im_num: int = 0 , vmin:float = 0, vmax: float = 1):
    fig, ax = plt.subplots(1,4, figsize = (30, 28))
    im1 = ax[0].imshow(im_dict[im_num], cmap = 'gray',)
    im2 = ax[1].imshow(tensor_dict1[im_num][0][0].numpy(), cmap = 'gray', vmin = vmin, vmax = vmax)
    im3 = ax[2].imshow(tensor_dict2[im_num][0][0].numpy(), cmap = 'gray', vmin = vmin, vmax = vmax)
    im4 = ax[3].imshow(target_dict[im_num], cmap = 'gray', )
    ax[0].set_title('Orig')
    ax[1].set_title('Independent')
    ax[2].set_title('Dependent')
    ax[3].set_title('GT')
    fig.show()
    print(f'Min: I:{torch.min(tensor_dict1[im_num][0][0]).item():7.5f} D: {torch.max(tensor_dict2[im_num][0][0]).item():7.5f}', )
    print(f'Min: I:{torch.max(tensor_dict1[im_num][0][0]).item():7.5f} D: {torch.max(tensor_dict2[im_num][0][0]).item():7.5f}', )

# %%
def display_tensor(tensor_dict:dict, im_dict:dict, target_dict:dict, im_num: int = 0 , vmin:float = 0, vmax: float = 1):
    fig, ax = plt.subplots(1,3, figsize = (18, 12))
    im1 = ax[0].imshow(im_dict[im_num], cmap = 'gray',)
    im2 = ax[1].imshow(tensor_dict[im_num][0][0].numpy(), cmap = 'gray', vmin = vmin, vmax = vmax)
    im3 = ax[2].imshow(target_dict[im_num], cmap = 'gray', )
    ax[0].set_title('Orig')
    ax[1].set_title('Expected')
    ax[2].set_title('GT')
    fig.show()
    print('Min: ' ,torch.min(tensor_dict[im_num][0][0]).item())
    print('Max: ' ,torch.max(tensor_dict[im_num][0][0]).item())

# %%
def extract_tensors(path, tensor_name):
    '''extracts tensors from path (as created when any uncertainty test is ran'''
    # get only the correct folders
    folders = []
    for sub_path in os.listdir(path):
        if sub_path.startswith('image'):
            folders.append( sub_path)
    
    # extract tensors
    all_images = {}
    for im_folder in folders:
        im_path = join(path, im_folder)
        # save the tensors in their own dict according to name before
        for tensor in os.listdir(im_path):
            if tensor == tensor_name:
                tensor_path = join(im_path, tensor)
                all_images[int(im_folder.split('_')[-1])] = torch.load(tensor_path, map_location = 'cpu')
    return all_images


# %%

def visualize_magnitudes(
                tensor_dicts,
                im_dict,
                target_dict,
                im_num = 0,
                vmin = 0,
                vmax = 1):
        arr = len(tensor_dicts) + 2

        cols = 4
        rows = math.ceil(arr / 4)

        fig, axes = plt.subplots(rows, cols, figsize = (8 * cols, 8*rows))

        axes = axes.flatten()

        axes[0].imshow(im_dict[im_num], cmap = 'gray',)
        axes[0].set_title('Input Image')
        for model_id, (model_name, model_dict) in enumerate(tensor_dicts.items()):
                model_id = model_id + 1
                axes[model_id].imshow(model_dict[im_num][0][0].numpy(), cmap = 'gray', vmin = vmin, vmax = vmax)
                axes[model_id].set_title(model_name)
        axes[-1].imshow(target_dict[im_num], cmap = 'gray',)
        axes[-1].set_title('GT Image')
        
        fig.show()

# %%
def calculate_magnitudes(std_dicts, mean_dicts):

        cols = ['model_name', 'im_num', 'min', 'max','mean', 'std',  'mean_thresholded_0.01', 'std_thresholded_0.01','mean_thresholded_0.001','std_thresholded_0.001','mean_thresholded_0','std_thresholded_0',]
        data = dict(zip(cols, [[] for i in range(len(cols))]))
        data2 = dict(zip(cols, [[] for i in range(len(cols))]))

        for model_id, (model_name, model_dict) in enumerate(std_dicts.items()):
                for im_num in model_dict:
                        
                        std_data = model_dict[im_num][0][0].flatten()
                        data['model_name'].append(model_name)
                        data['im_num'].append(im_num)
                        data['min'].append(torch.min(std_data).item())
                        data['max'].append(torch.max(std_data).item())
                        data['mean'].append(torch.mean(std_data).item())
                        data['std'].append(torch.std(std_data).item())
                        data['mean_thresholded_0.01'].append(torch.mean(std_data[std_data > 0.01]).item())
                        data['std_thresholded_0.01'].append(torch.std(std_data[std_data > 0.01]).item())
                        data['mean_thresholded_0.001'].append(torch.mean(std_data[std_data > 0.001]).item())
                        data['std_thresholded_0.001'].append(torch.std(std_data[std_data > 0.001]).item())
                        data['mean_thresholded_0'].append(torch.mean(std_data[std_data > 0]).item())
                        data['std_thresholded_0'].append(torch.std(std_data[std_data > 0]).item())

        for model_id, (model_name, model_dict) in enumerate(mean_dicts.items()):
                for im_num in model_dict:
                        mean_data = model_dict[im_num][0][0].flatten()
                        data2['model_name'].append(model_name)
                        data2['im_num'].append(im_num)
                        data2['min'].append(torch.min(mean_data).item())
                        data2['max'].append(torch.max(mean_data).item())
                        data2['mean'].append(torch.mean(mean_data).item())
                        data2['std'].append(torch.std(mean_data).item())
                        data2['mean_thresholded_0.01'].append(torch.mean(mean_data[mean_data > 0.01]).item())
                        data2['std_thresholded_0.01'].append(torch.std(mean_data[mean_data > 0.01]).item())
                        data2['mean_thresholded_0.001'].append(torch.mean(mean_data[mean_data > 0.001]).item())
                        data2['std_thresholded_0.001'].append(torch.std(mean_data[mean_data > 0.001]).item())
                        data2['mean_thresholded_0'].append(torch.mean(mean_data[mean_data > 0]).item())
                        data2['std_thresholded_0'].append(torch.std(mean_data[mean_data > 0]).item())
        
        return pd.DataFrame(data, columns = cols), pd.DataFrame(data2, columns = cols)

# %%
def display_std_hist(std_dict, key = 'BM-1', im_num = 0, threshold = .01):
    data = std_dict[key][im_num].flatten()
    data = data[data > threshold]
    print("Data size:", len(data))
    plt.hist(data, bins = 'auto')

# %%
def add_data(path, name, df):
    temp = pd.read_csv(path, index_col = 0)
    temp['name'] = name
    return pd.concat([df, temp])

# %%
def pseudo_anim(tensor_dict: dict, val_num: int = 0 , im_num: int = 0 ):
    ''' creates an animated gif using images
    
    output:
    creates a gif'''

    fig, ax = plt.subplots(1,1, figsize = (8, 8))
    ax.imshow(tensor_dict[val_num][im_num][0][0].numpy(), cmap = 'gray', vmin = 0, vmax = 1)
    fig.show()
    

# %%
def display_agg_std_hist(std_dict, key = 'BM-1', threshold = .01):
    data = torch.cat(list(std_dict[key].values())).flatten()
    data = data[data > threshold]
    print("Data size:", len(data))
    plt.hist(data, bins = 'auto', density = True)

def display_agg_CV_hist(std_dict, mean_dict, masks_dict, key = 'BM-1', range=(0,5), threshold = .01, name = 'DB', save = False, save_folder = None):
    
    std_data = []
    mean_data = []

    # remove mask
    for data, value in std_dict[key].items():
        seg_std = std_dict[key][data]
        seg_mean = mean_dict[key][data]
        mask = TF.resize(transforms.ToTensor()(~masks_dict[data]), (seg_std.shape[-2:]))[0].numpy()
        temp = np.ma.array(seg_std, mask = mask)
        temp2 = np.ma.array(seg_mean, mask = mask)
        std_data.append(temp[~temp.mask].data)
        mean_data.append(temp2[~temp2.mask].data)
    std_data = np.concatenate(std_data)
    mean_data =  np.concatenate(mean_data)
    cv_data = (std_data/mean_data)

    cv_data = cv_data.flatten()
    cv_data = cv_data[~np.isnan(cv_data)]

    

    #print(cv_data.max())
    #cv_data = cv_data[cv_data > threshold]
    #print("Data size:", len(cv_data))
    
    fig, ax = plt.subplots(1,1)
    
    ax.hist(cv_data, bins = 'auto', range = range, density = True)
    fig.suptitle(key + f' {name} CV')
    if save:
        fig.savefig(os.path.join(save_folder, f"CV_Histogram_{key}.png"))
        plt.close(fig)
    else:
        fig.show()
        
    

# %%
def dilated_agg_std_hist(data, targets, model, threshold, range, name, save, savefolder):
    aggdata = []
    for im_num, value in data[model].items():
        seg = data[model][im_num]
        target = TF.resize(transforms.ToTensor()(~targets[im_num]), (seg.shape[-2:]))[0].numpy()
        dilated = cv2.erode(target, np.ones((3,3),np.uint8))
        temp = np.ma.array(seg,mask = dilated)
        aggdata.append(temp[~temp.mask].data)
    aggdata = np.concatenate(aggdata)
    #data = data[data > threshold]
    fig, ax =  plt.subplots(1,1)
    ax.hist(aggdata, bins = 'auto', range = range, density = True)
    fig.suptitle(name)
    if save:
        fig.savefig(os.path.join(savefolder, f'STD_Dilated_Histogram_{model}.png'))
        plt.close(fig)
    else:
        fig.show()
def dilated_agg_cv_hist(data,mean, targets, model, threshold, range, name, save, savefolder):
    aggdata = []
    aggdata2 = []
    for im_num, value in data[model].items():
        seg = data[model][im_num]
        seg_mean = mean[model][im_num]
        target = TF.resize(transforms.ToTensor()(~targets[im_num]), (seg.shape[-2:]))[0].numpy()
        dilated = cv2.erode(target, np.ones((3,3),np.uint8))
        temp = np.ma.array(seg,mask = dilated)
        temp2 = np.ma.array(seg_mean, mask = dilated)
        aggdata.append(temp[~temp.mask].data)
        aggdata2.append(temp2[~temp2.mask].data)
    aggdata = np.concatenate(aggdata)
    aggdata2 = np.concatenate(aggdata2)
    aggdata[aggdata2 == 0] = 1e-8
    aggdata2[aggdata2 == 0] = 1e-8
    
    aggdata = aggdata/aggdata2
    
    #aggdata = aggdata[aggdata > threshold]
    
    fig, ax =  plt.subplots(1,1)
    ax.hist(aggdata, bins = 'auto', range = range, density = True)
    fig.suptitle(name)
    if save:
        fig.savefig(os.path.join(savefolder, f'CV_Dilated_Histogram_{model}.png'))
        plt.close(fig)
    else:
        fig.show()

# %%
def inv_dilated_agg_std_hist(data, targets, masks, model, threshold, range, name, save, savefolder,):
    aggdata = []
    for im_num, value in data[model].items():
        seg = data[model][im_num]
        target = TF.resize(transforms.ToTensor()(~targets[im_num]), (seg.shape[-2:]))[0].numpy()
        mask = TF.resize(transforms.ToTensor()(masks[im_num]), (seg.shape[-2:]))[0].numpy()
        dilated = cv2.erode(target, np.ones((3,3),np.uint8))
        
        inv_dilated = 1 - (mask * dilated)
        temp = np.ma.array(seg ,mask = inv_dilated)
        aggdata.append(temp[~temp.mask].data)
    aggdata = np.concatenate(aggdata)
    #data = data[data > threshold]
    fig, ax =  plt.subplots(1,1)
    ax.hist(aggdata, bins = 'auto', range = range, density = True)
    fig.suptitle(name)
    if save:
        fig.savefig(os.path.join(savefolder, f'STD_Dilated_Histogram_{model}.png'))
        plt.close(fig)
    else:
        fig.show()
def inv_dilated_agg_cv_hist(data,mean, targets,masks, model, threshold, range, name, save, savefolder,):
    aggdata = []
    aggdata2 = []
    for im_num, value in data[model].items():
        seg = data[model][im_num][0][0]
        seg_mean = mean[model][im_num]
        target = TF.resize(transforms.ToTensor()(~targets[im_num]), (seg.shape[-2:]))[0].numpy()
        mask = TF.resize(transforms.ToTensor()(masks[im_num]), (seg.shape[-2:]))[0].numpy()
        dilated = cv2.erode(target, np.ones((3,3),np.uint8))
        inv_dilated = 1 - (mask * dilated)
        temp = np.ma.array(seg,mask = inv_dilated)
        temp2 = np.ma.array(seg_mean, mask = inv_dilated)
        aggdata.append(temp[~temp.mask].data)
        aggdata2.append(temp2[~temp2.mask].data)
    aggdata = np.concatenate(aggdata)
    aggdata2 = np.concatenate(aggdata2)
    aggdata[aggdata2 == 0] = 1e-8
    aggdata2[aggdata2 == 0] = 1e-8
    aggdata = aggdata/aggdata2
    
    #aggdata = aggdata[aggdata > threshold]
    
    fig, ax =  plt.subplots(1,1)
    ax.hist(aggdata, bins = 'auto', range = range, density = True)
    fig.suptitle(name)
    if save:
        fig.savefig(os.path.join(savefolder, f'CV_Dilated_Histogram_{model}.png'))
        plt.close(fig)
    else:
        fig.show()

# %%


# %%

import matplotlib.colors as colors
def seg_to_im(n, norm = False, cmap = 'jet', reverse = False):
    im = transforms.ToPILImage()(n)
    im = np.array(im)
    cm = plt.get_cmap(cmap)
    cm.set_over('k') # set over to black
    if reverse:
        cm = cm.reversed()
    if norm:
        im = colors.Normalize()(im)
    colored_image = cm(im)
    return Image.fromarray((colored_image[:, :, :3]  * 255).astype(np.uint8))
    

# %%


# %% [markdown]
# # Validation Images

# %%
val_root_images = 'augmented_data/val/images'
orig = {}
# retrieves validation images as grayscale
for im in os.listdir(val_root_images):
    orig[int(im.split('_')[0])] = np.array(Image.open(join(val_root_images, im)).convert('L'))

val_root_targets = 'augmented_data/val/targets'
targets = {}
for im in os.listdir(val_root_targets):
    targets[int(im.split('_')[0])] = np.array(Image.open(join(val_root_targets, im)).convert('L'))

mask_root_targets = 'augmented_data/val/masks'
masks = {}
for im in os.listdir(mask_root_targets):
    masks[int(im.split('_')[0])] = np.array(Image.open(join(mask_root_targets, im)).convert('L'))


# %% [markdown]
# # Viewing and Saving Plots

# %% [markdown]
# retrieve data

# %%
colorscheme = {'BM-1':'tab:blue', 'BM-2' : 'tab:blue', 'BM-3': 'tab:blue', 'LF-1': 'tab:orange', 'LF-3':'tab:orange', 'LF-5': 'tab:orange', 'LF-2': 'tab:green', 'LF-4': 'tab:green', 'LF-6': 'tab:green', 'MF-1': 'tab:red', 'MF-2': 'tab:red', 'MF-3':'tab:red'}
markerscheme = {'BM-1':'-', 'BM-2' : ':', 'BM-3': '--', 'LF-1': '-.', 'LF-3':':', 'LF-5': '--', 'LF-2': '-', 'LF-4': ':', 'LF-6': '--', 'MF-1': '-', 'MF-2': ':', 'MF-3':'--'}


# %%
colorscheme_im = {0: 'tab:blue', 1: 'tab:orange',2: 'tab:green', 3: 'tab:red', 4: 'tab:purple', 5:'tab:brown'}

# %%
all_data = pd.DataFrame()
mean_db_tensors = {}
std_db_tensors = {}
mean_rot_tensors = {}
std_rot_tensors = {}
seg_tensors = {}
models = 'BM-1 BM-2 BM-3 MF-1 MF-2 MF-3 LF-1 LF-3 LF-5 LF-2 LF-4 LF-6'.split()
base_models = 'BM-1 BM-2 BM-3'.split()
mf_models = 'MF-1 MF-2 MF-3'.split()
LF_HFT_models =  'LF-1 LF-3 LF-5'.split()
LF_LFT_models = 'LF-2 LF-4 LF-6'.split()

# %%
drive_root = 'results/DRIVE/'
#models
for model_id in models:
    path = join(drive_root, model_id)
    # reg
    all_data = add_data(join(path, 'statistics/val_images/metrics.csv'), name = model_id, df =  all_data)
    # db
    all_data = add_data(join(path, 'dropblock_uncertainty/statistics/val_images/metrics.csv'), name = f'{model_id}_DB', df = all_data)
    seg_tensors[model_id] = extract_tensors(join(path, 'statistics/val_images/tensors'), 'segmentation.pt')
    mean_db_tensors[model_id] = extract_tensors(join(path, 'dropblock_uncertainty/tensors'), 'mean.pt')
    std_db_tensors[model_id] = extract_tensors(join(path, 'dropblock_uncertainty/tensors'), 'std.pt')
    mean_rot_tensors[model_id] = extract_tensors(join(path, 'rotation_uncertainty'), 'mean.pt')
    std_rot_tensors[model_id] = extract_tensors(join(path, 'rotation_uncertainty'), 'std.pt')

# %%
def std_density(models, std_data, threshold, rnge, num_steps, figname, xlabel, ylabel, save, save_path, colorscheme, markerscheme):
    # overlapping std DB density
    bandwidth = (rnge[1] - rnge[0])/num_steps # bandwidth to approx density, must be small b/c our data lies in a small range
    fig, ax = plt.subplots(1, 1, figsize = (15, 10))
    for model in models:
        #if len(save_info) == len(models):
        #    X_plot = np.linspace(rnge[0], rnge[1], num_steps)[:, np.newaxis]
        #    ax.plot(X_plot[:, 0], np.exp(save_info[model]), markerscheme[model],c = colorscheme[model], label = model , alpha = .5, linewidth = 5)
        #else:
        print(model)
        data = []
        for i in range(len(std_data[model])):
            data.append(std_data[model][i].flatten().numpy()) # combine all stddevs
        data = np.concatenate(data)
        data = data[data>threshold][:, np.newaxis]
        kde = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(data)
        X_plot = np.linspace(rnge[0], rnge[1], num_steps)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        #save_info[model] = log_dens
        ax.plot(X_plot[:, 0], np.exp(log_dens), markerscheme[model],c = colorscheme[model], label = model , alpha = .6, linewidth = 1.5)

    leg = ax.legend(ncol = 4, frameon = False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(figname)
    figname = '_'.join(figname.split(' '))
    if save:
        fig.savefig(os.path.join(save_path, f'{figname}.png'))
        plt.close(fig)
    else:
        fig.show()
        

# %%
threshold = .01 # to remain consistent
rnge = (0, .5)
num_steps = 1000

figtitle = "All Model DB STD"
save_path = 'results/Images/All_Models'
std_density(models, std_data = std_db_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle, xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

figtitle = "Base Model DB STD"
save_path = 'results/Images/All_Models'
std_density( ['BM-1', 'BM-2', 'BM-3'], std_data = std_db_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle,xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

figtitle = "LF HFT Model DB STD"
save_path = 'results/Images/All_Models'
std_density( ['LF-1', 'LF-3', 'LF-5'], std_data = std_db_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle,xlabel = 'STD', ylabel = 'Density',  save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

figtitle = "LF LFT Model DB STD"
save_path = 'results/Images/All_Models'
std_density( ['LF-2', 'LF-4', 'LF-6'], std_data = std_db_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle,xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

figtitle = "Multi Fidelity Model DB STD"
save_path = 'results/Images/All_Models'
std_density( ['MF-1', 'MF-2', 'MF-3'], std_data = std_db_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle, xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

# %%
threshold = .01 # to remain consistent
rnge = (0, .3)
num_steps = 1000
figtitle = "All Model ROT STD"
save_path = 'results/Images/All_Models'
std_density(models, std_data = std_rot_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle, xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)


figtitle = "Base Model ROT STD"
save_path = 'results/Images/All_Models'
std_density( ['BM-1', 'BM-2', 'BM-3'], std_data = std_rot_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle,xlabel = 'STD', ylabel = 'Density',  save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

figtitle = "LF HFT Model ROT STD"
save_path = 'results/Images/All_Models'
std_density( ['LF-1', 'LF-3', 'LF-5'], std_data = std_rot_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle, xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

figtitle = "LF LFT Model ROT STD"
save_path = 'results/Images/All_Models'
std_density( ['LF-2', 'LF-4', 'LF-6'], std_data = std_rot_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle, xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

figtitle = "Multi Fidelity Model ROT STD"
save_path = 'results/Images/All_Models'
std_density( ['MF-1', 'MF-2', 'MF-3'], std_data = std_rot_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle, xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme, markerscheme = markerscheme)

# %%
   # Single Model     
def std_single_density(model, std_data, threshold, rnge, num_steps, figname, xlabel, ylabel, save, save_path, colorscheme):
    bandwidth = (rnge[1] - rnge[0])/num_steps # bandwidth to approx density, must be small b/c our data lies in a small range
    fig, ax = plt.subplots(1, 1, figsize = (15, 10))
    
    print(model)
    for im, data in std_data[model].items():
        data = data.flatten().numpy() 
        data = data[data>threshold][:, np.newaxis]
        kde = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(data)
        X_plot = np.linspace(rnge[0], rnge[1], num_steps)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        ax.plot(X_plot[:, 0], np.exp(log_dens),c = colorscheme[im], label = im , alpha = .6, linewidth = 1.5)

    leg = ax.legend(ncol = 2, frameon = False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(figname)
    figname = '_'.join(figname.split(' '))
    if save:
        fig.savefig(os.path.join(save_path, f'{figname}.png'))
        plt.close(fig)
    else:
        fig.show()
             


# %%
threshold = .01 # to remain consistent
num_steps = 250

for model in models:
    figtitle = f"{model} DB STD"
    rnge = (0, .5)
    save_path = 'results/Images/Single_Models'
    std_single_density(model, std_data = std_db_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle, xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme_im)
    
    figtitle = f"{model} ROT STD"
    rnge = (0, .3)
    save_path = 'results/Images/Single_Models'
    std_single_density(model, std_data = std_rot_tensors, threshold = threshold, rnge = rnge, num_steps = num_steps, figname = figtitle, xlabel = 'STD', ylabel = 'Density', save = True, save_path = save_path, colorscheme = colorscheme_im,)
