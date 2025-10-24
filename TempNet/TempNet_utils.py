
"""
Created on June 12th, 2025

@author: Lexie Hassien
"""

import os
import glob
import time
#import subprocess
import numpy as np 
import random
import pandas as pd
#import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
#import matplotlib.colors as clr
#import seaborn as sns
import torch
# import argparse
# import math

## Set seed for reproducibility 
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


## Load count and meta data
def load_data(data_dir, predictor_key, temporal_key, filter_key = None):
            
    ## Load in the gene expression counts and meta data 
    # each sample should have a .meta.csv file and a .cnt.csv file
    meta_file_list = sorted(glob.glob(os.path.join(data_dir, '*.meta.csv')))

    t0 = time.time()
    data_frames = []
    label_frames = []
    for meta_path in meta_file_list:
        print(meta_path)
        cnt_path = meta_path.replace(".meta.csv", ".cnt.csv")
        meta = pd.read_csv(meta_path, sep=",", header=0, na_filter=False, index_col=0) # index = cell ID
        cnt = pd.read_csv(cnt_path,  sep=",", header=0, na_filter=False, index_col=0) # index = cell ID

        # organize and append dataframes in list
        if filter_key != None:
            labels = pd.DataFrame({filter_key: meta[filter_key], predictor_key: meta[predictor_key], temporal_key: pd.Series(meta[temporal_key], dtype='string')})
        else:
            labels = pd.DataFrame({predictor_key: meta[predictor_key], temporal_key: pd.Series(meta[temporal_key], dtype='string')})
        data = cnt.transpose()  # rows = cells, cols = genes
        data_frames.append(data)
        label_frames.append(labels)

    df_data_comb = pd.concat(data_frames, join='inner')
    df_label_comb = pd.concat(label_frames, join='inner')
    print(f'Time elapsed for loading data: {str(round((time.time()-t0)/60,3))} minutes')
    print(f'Number of cells: {df_label_comb.shape[0]}')
    
    return df_data_comb, df_label_comb

## Pre-process steps if necessary
def sc_preprocess(adata_obj):
    sc.pp.normalize_total(adata_obj)
    sc.pp.log1p(adata_obj)
    sc.pp.highly_variable_genes(adata_obj, n_top_genes=2000)
    sc.pp.scale(adata_obj, max_value=10)
    sc.tl.pca(adata_obj, n_comps=50, svd_solver=None, mask_var = "highly_variable")
    return adata_obj

### Maximum Mean Discrepancy (MMD)
def MMD_sigmas(x, y, device, sigmas=0):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx 
    dyy = ry.t() + ry - 2. * yy 
    dxy = rx.t() + ry - 2. * zz
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if sigmas == 0:
        a = x.cpu().detach().numpy()
        b = y.cpu().detach().numpy()
        mat = np.concatenate((a,b))
        dist = []
        nsamp = mat.shape[0]
        for i in range(nsamp):
            euc_dist = np.sqrt(np.sum(np.square(np.subtract(mat[i,:], mat)), axis=1))
            dt = np.array(sorted(euc_dist))
            dist.append(dt[dt>0][0])
   
        const = [1,2,4,8,16]
        for c in const:
            sigma = np.square(c*np.median(dist))
            sigmas = torch.FloatTensor([sigma]).to(device)     
            XX += torch.exp(-0.5*dxx/sigmas)
            YY += torch.exp(-0.5*dyy/sigmas)
            XY += torch.exp(-0.5*dxy/sigmas)  
    else:
            XX += torch.exp(-0.5*dxx/sigmas)
            YY += torch.exp(-0.5*dyy/sigmas)
            XY += torch.exp(-0.5*dxy/sigmas) 
    return torch.mean(XX + YY - 2. * XY)


## UMAP Visualization
def plot_umap(adata_obj, predictor_key, predictor_labels, temporal_key, temporal_labels, seed = 0, save_dir = None):
    
    ## Parameters for UMAP visualization
    n_neighbors = 40
    min_dist = 0.8
    neighbors_metric = 'cosine' # see justification: https://link.springer.com/chapter/10.1007/3-540-44503-X_27
    sc.pp.neighbors(adata_obj, use_rep = "X", n_neighbors = n_neighbors, metric = neighbors_metric)
    sc.tl.umap(adata_obj, min_dist = min_dist, init_pos = "random",random_state=seed)

    ## Specify color maps for predictor and temporal labels
    ## Use color-blind palette for accessibility
    predictor_colors = ['#b6dbff','#db6d00','#009292', '#ff6db6', '#b66dff',  '#490092','gray', '#920000','#ffb6db',
                        '#24ff24','#004949','#006ddb','#ffff6d']
    predictor_cmap = dict(map(lambda i,j : (i,j) , predictor_labels, predictor_colors))
    adata_obj.obs[f"{predictor_key}_ordered"] = pd.Categorical(
        values=adata_obj.obs[predictor_key].copy(), categories=predictor_labels, ordered=True)

    if len(temporal_labels) <= 4:
        temporal_colors =  ['#440154ff', '#33638dff','#3cbb75ff', '#fde725ff']
    else:
        temporal_colors = ['#6DE9EE','#4083E7', '#D5FD85', '#95EB48','#8E71D5','#BB27C6','#DF7D2E','#DA2F20'] # sequential color palette
    temporal_cmap = dict(map(lambda i,j : (i,j) , temporal_labels, temporal_colors))
    adata_obj.obs[f"{temporal_key}_ordered"] = pd.Categorical(
        values=adata_obj.obs[temporal_key].copy(), categories=temporal_labels, ordered=True)
    
    ## Initialize figure
    plt.rcParams['font.size'] = 12.0
    fig, axes = plt.subplots(1, 2, figsize=(11,4))

    # for aesthetic purposes, can use random re-ordering of indices (doesn't change the shape of the plot, just which data points are "on top")
    # https://scanpy-tutorials.readthedocs.io/en/latest/plotting/advanced.html#cell-ordering
    rng = np.random.default_rng(seed = seed)
    random_indices = rng.permutation(list(range(adata_obj.shape[0])))

    ## Use scanpy and matplotlib functions to plot UMAP and save
    point_sz = 15
    sc.pl.umap(adata_obj[random_indices,:].copy(), color=predictor_key, ax = axes[0], palette = predictor_cmap, show = False, 
            size = point_sz, title = f'Embedding by {predictor_key}')
    sc.pl.umap(adata_obj[random_indices,:].copy(), color=temporal_key, ax = axes[1], palette = temporal_cmap, show = False, 
            size = point_sz, title = f'Embedding by {temporal_key}')
    
    plt.tight_layout()
    if save_dir != None:
        plt.savefig(save_dir + f"/TempNet_embed.png")

