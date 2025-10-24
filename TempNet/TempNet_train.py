
"""
Created on June 12th, 2025
@author: Lexie Hassien
"""

import os
import argparse
import numpy as np 
import pandas as pd
import gc
import time
import anndata as ad
import scanpy as sc
# import scanpy.external as sce
# import matplotlib.pyplot as plt
# import matplotlib.colors as clr
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
#from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import chain
#from IPython.display import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import DataLoader

from TempNet_utils import *
from TempNet_submodels import *


# ==============================================================================
# =                                Input arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Define a custom argument type for a list of strings
def list_of_strings(arg):
    arg = arg.split(',') 
    return list(map(str, arg))
parser.add_argument('--data_dir', type=str, default="./data/sample_data", help='Directory where data files are located') 
                    # Each sample should a count file (sample1.cnt.csv) and a metadata (sample1.meta.csv) file
                    # cnt.csv files have rows = genes and columns = cells
parser.add_argument('--temporal_key', type=str, default="stage", help='Column name representing temporal stage (e.g., stage, age, grade)')
parser.add_argument('--temporal_labels', type=list_of_strings) 
                    # Temporal labels must be listed in sequential order 
parser.add_argument('--predictor_key', type=str, default="cell_type", help='Column name for characteristic used in the predictor (e.g., cell type, sample)')
parser.add_argument('--filter_key', type=str, default = None, help='Categorical variable for selecting subset of data (e.g., dissection_part)')
parser.add_argument('--filter_labels', type=list_of_strings, help = 'Categorical labels to include within filter_key category (e.g., head)')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
parser.add_argument("--do_preprocess",action = "store_true",help="Whether to pre-process the data using standard Scanpy pipeline")
parser.add_argument('--n_blocks', type=int, default=2, help='Number of ResNet blocks')
parser.add_argument('--embed_dim', type=int, default=128, help='Dimension of latent space embedding')
parser.add_argument('--batch_size', type=int, default=400, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning_rate')
parser.add_argument("--num_epochs",type=int,default=20, help="Number of epochs")
parser.add_argument("--mmd_loss_w",type=float,default=0.7,help="Weight of MMD loss in encoder")
parser.add_argument("--cosine_loss_w",type=float,default=0.3,help="Weight of cosine embedding loss in encoder")
parser.add_argument("--enc_loss_w",type=float, default=0.8, help="Weight of encoder loss in full model") 
parser.add_argument("--disc_loss_w",type=float,default=0.4,help="Weight of discriminator loss in full model")
parser.add_argument("--pred_loss_w",type=float,default=1,help="Weight of predictor loss in full model")
parser.add_argument("--use_rev",action = "store_true",default=False,help="Whether to use gradient reversal")
parser.add_argument("--rev_alpha",type=float,default=1,help="Gradient reversal level: between 0 and 1")
parser.add_argument("--save_dir",type=str,default='./save_dir', help="Directory for saving TempNet embedding")
parser.add_argument("--save_model",action = "store_true",help="Whether to save the model for reproducibility")
args = parser.parse_args()

# ==============================================================================
# =                      Device set up                                         =
# ==============================================================================

## GPU device configuration
dev_num = 0
cuda = torch.cuda.is_available()
device = torch.device(f'cuda:{dev_num}' if cuda else 'cpu')
print('Is GPU available? ' + str(cuda))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if cuda:
    torch.cuda.set_device(dev_num)
    print(torch.cuda.current_device())

## Set seed for reproducibility
seed_torch(args.seed) # function in utils
rng = np.random.default_rng(seed = args.seed)

# ==============================================================================
# =                       Load and organize data                               =
# ==============================================================================

## Load in the gene expression counts and meta data
# Each sample should a count file (sample1.cnt.csv) and a metadata (sample1.meta.csv) file
# cnt.csv files have rows = genes and columns = cells, .meta.csv files should have rows = cells, columns = metadata categories
print("Loading data...")
df_data_comb, df_label_comb = load_data(args.data_dir, args.predictor_key, args.temporal_key, args.filter_key)
predictor_labels = np.unique(df_label_comb[args.predictor_key])

# List of cell types to include
if args.filter_key != None:
    df_data_comb = df_data_comb[df_label_comb[args.filter_key].isin(args.filter_labels)]
    df_label_comb = df_label_comb[df_label_comb[args.filter_key].isin(args.filter_labels)]
    print(f'Number of cells after filtering: {df_label_comb.shape[0]}')

## Ensure combined data (counts and labels) are ordered according to their temporal label
df_label_comb[f"{args.temporal_key}_cat"] = pd.Categorical(df_label_comb[args.temporal_key], categories= args.temporal_labels, ordered=True)
df_label_comb = df_label_comb.sort_values(by=[f"{args.temporal_key}_cat"])
df_data_comb = df_data_comb.reindex(df_label_comb.index)

## Create AnnData object
adata_comb = sc.AnnData(df_data_comb)
adata_comb.obs[args.predictor_key] = df_label_comb[args.predictor_key].values
adata_comb.obs[args.temporal_key] = df_label_comb[args.temporal_key].values
adata_comb.obs[f"{args.temporal_key}_ordered"] = pd.Categorical(
    values=adata_comb.obs[args.temporal_key], categories=args.temporal_labels, ordered=True)

# Pre-processing steps
# Apply normalization and transformation if necessary
if args.do_preprocess == True:
    adata_comb = sc_preprocess(adata_comb)

# ==============================================================================
# =                       Prepare data for model                               =
# ==============================================================================

print("Preparing data for model...")

## Create array of the number of observations (cells=n) for each time point
# [n1, n2, n3 ...]
df_label_comb = adata_comb.obs
group = df_label_comb.groupby(f"{args.temporal_key}_ordered").count() # count the observations 
szmat = group.iloc[:,0].values

## Prepare label variables
df_label_comb[f'{args.temporal_key}_cat'] = df_label_comb[args.temporal_key].astype('category')
df_label_comb[f'{args.temporal_key}_codes'] = df_label_comb[f'{args.temporal_key}_cat'].cat.codes
df_label_comb[f'{args.predictor_key}_cat'] = df_label_comb[args.predictor_key].astype('category')
df_label_comb[f'{args.predictor_key}_codes'] = df_label_comb[f'{args.predictor_key}_cat'].cat.codes

## Create tensors for output labels for predictor and discriminator
out_tm = torch.LongTensor(np.array(df_label_comb[f'{args.temporal_key}_codes'].values))
out_ty = torch.LongTensor(np.array(df_label_comb[f'{args.predictor_key}_codes'].values))

## Format data and create torch tensors
df_data_comb = adata_comb.to_df()

# ## Apply PCA
# pca = PCA(n_components=50)  
# X_reduced = pca.fit_transform(df_data_comb)
# df_data_comb = pd.DataFrame(X_reduced)
# print(df_data_comb.shape)

## Format training data
train_data = torch.FloatTensor(np.array(df_data_comb))
train_data = train_data[:,None,:]

## Create list of indexes that separate each individual within the combined data
## [0, n1, n1+n2, n1+n2+n3 ...]
szmat2 = list() 
nt = len(szmat) # number of individuals
for i in range(nt):
    szmat2 += [str(np.sum(szmat[0:i]))]
szmat2 = np.array(szmat2).astype(int)


# ==============================================================================
# =                               Initialize models                            =
# ==============================================================================

print("Initializing models...")

## Define models and specify model parameters
in_dim = df_data_comb.shape[1] # number of features
n_classes = len(np.unique(df_label_comb[f'{args.predictor_key}_codes']))# number of layers/cell_types for classification
n_tm = len(np.unique(df_label_comb[f'{args.temporal_key}_codes']))# number of layers/cell_types for classification

## Initialize models
encoder = ResNet1DEncoder2(
        in_dim=in_dim,
        embed_dim=args.embed_dim,
        base_filters=1, 
        kernel_size=round(args.batch_size*(2/3)), 
        dilation=1,
        stride=1, 
        n_block=args.n_blocks, 
        groups=1,
        n_classes=n_classes, 
        downsample_gap=1, 
        dropout = 0.4,
        use_bn=True,
        use_do = True,
        verbose=False)
discriminator = Discriminator(args.embed_dim, n_tm)
predictor = Predictor(args.embed_dim, n_classes, args.use_rev, args.rev_alpha) 

## Add models to device
encoder = encoder.to(device)
discriminator = discriminator.to(device)
predictor = predictor.to(device)

## Specify loss functions
class_loss = nn.CrossEntropyLoss()
adversarial_loss = nn.CrossEntropyLoss() #nn.BCELoss() for binary case
cosine_loss = nn.CosineEmbeddingLoss()

# Define optimizers
enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
pred_optimizer = torch.optim.Adam(chain(encoder.parameters(), predictor.parameters()), lr=args.learning_rate)
disc_optimizer = torch.optim.Adam(chain(encoder.parameters(), discriminator.parameters()), lr=args.learning_rate)

# ==============================================================================
# =                               Model training                               =
# ==============================================================================

print("Starting model training...")

num_stages = len(args.temporal_labels)
totalsz=df_data_comb.shape[0]
totalbatsz=args.batch_size*num_stages

# Reverse gradients, if applicable
if args.use_rev ==True:
    rev_alpha = torch.tensor([args.rev_alpha]) # reverse level, between [0,1]
    if cuda:
        rev_alpha = rev_alpha.cuda()

## Training loop
t00 = time.time() # start time for training loop
for epoch in range(args.num_epochs):
    train_acc = []
    t0 = time.time() # start time for each epoch
    predictor.train() 
    encoder.train()

    for dat in range(20): 
        # Select a random batch for training feature encoder and classifier
        idx_perm = np.random.permutation(szmat[0])
        idx0 = idx_perm[:int(args.batch_size)]
        idx = idx0
        for k in range(1,num_stages):
            idx_perm = np.random.permutation(szmat[k])
            idx0 = idx_perm[:int(args.batch_size)]
            idx = np.concatenate((idx,idx0+int(szmat2[k])))
        
        x_input = train_data[idx]
        gt_labs = out_tm[idx] # batch of temporal labels
        gt_classes = out_ty[idx] # batch of predictor labels
        if cuda:
            x_input = x_input.cuda()
            gt_labs = gt_labs.cuda()
            gt_classes = gt_classes.cuda()

        ######################################################## Encoder loss
        x_emb = encoder(x_input)
        sz = int(x_emb.size()[0]/num_stages)
        embs = torch.split(x_emb, sz)
        n=len(embs)-1
        train_mmd_loss = 0
        train_cosine_loss = 0

        ## Compare the embeddings sequentially 
        for i in range(n):
            x1_emb = embs[i]
            x2_emb = embs[i+1]
            train_mmd_loss += MMD_sigmas(x1_emb, x2_emb, device, sigmas = 0)
            cosine_y = torch.ones(x1_emb.shape[0])
            if cuda:
                cosine_y = cosine_y.cuda()
            train_cosine_loss += cosine_loss(x1_emb, x2_emb, cosine_y)
        train_enc_loss = args.mmd_loss_w*train_mmd_loss + args.cosine_loss_w*train_cosine_loss 

        ######################################################## Discriminator loss
        dm_z = discriminator(x_emb) 
        train_disc_loss = adversarial_loss(dm_z, gt_labs)

        ######################################################### Predictor loss
        classes = predictor(x_emb)
        train_pred_loss = class_loss(classes, gt_classes) 

        # Total training loss
        train_loss = args.enc_loss_w*train_enc_loss + args.disc_loss_w*train_disc_loss + args.pred_loss_w*train_pred_loss 

        # Backward and optimize for resnet feature extraction model
        enc_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        pred_optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        train_loss.backward()
        
        enc_optimizer.step()
        disc_optimizer.step()
        pred_optimizer.step()
        torch.cuda.empty_cache()
        gc.collect()

        # Calculate training accuracy for predicting class labels
        if cuda:
            y_true = gt_classes.cpu().detach()
        else:
            y_true = gt_classes.detach()
        classes = F.softmax(classes, dim=1)
        y_pred = torch.argmax(classes, dim=1)
        train_acc.append(accuracy_score(y_true, y_pred.cpu()))
    train_acc_mean = np.mean(train_acc)  

    ########### TEST SET ##########
    predictor.eval() 
    encoder.eval()
    with torch.no_grad():
        for dat in range(20):
            idx_perm = np.random.permutation(int(totalsz))
            idx = idx_perm[:int(totalbatsz)]
            x_input = train_data[idx]
            gt_classes = out_ty[idx]
            if cuda:
                x_input = x_input.cuda()
                gt_classes = gt_classes.cuda()
            
            # encoder and predictor
            x_emb = encoder(x_input)
            classes = predictor(x_emb)
    
            # test accuracy
            if cuda:
                y_true = gt_classes.cpu().detach()
            else:
                y_true = gt_classes.detach()
            classes = F.softmax(classes, dim=1)
            y_pred = torch.argmax(classes, dim=1)
             
   
    print('Epoch [{}/{}], Encoder Loss: {:.5f}, Predictor Loss: {:.5f}, Discriminator Loss: {:.5f}, Train Accuracy: {:.4f}' 
                   .format(epoch, args.num_epochs, train_enc_loss.item(), train_pred_loss.item(), train_disc_loss.item(), train_acc_mean))
    print('Training time for this epoch: ' + str(round(time.time()-t0,3)) + ' seconds \n')

## Total run time 
print('Training complete')
print(f'Total training time: {str(round((time.time()-t00)/60,3))} minutes')

# ==============================================================================
# =                               Model evaluation                             =
# ==============================================================================

print("Processing model results...")

# ## Save the trained model
if args.save_model == True:
    torch.save(encoder.state_dict(), args.save_dir + 'encoder_state_dict.pt')
    torch.save(predictor.state_dict(), args.save_dir + 'predictor_state_dict.pt')
    torch.save(encoder, args.save_dir + 'encoder_pytorch.pt')
    torch.save(predictor, args.save_dir + 'predictor_pytorch.pt')

## Extract embedding from trained encoder
# register hook
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
encoder.final_pool.register_forward_hook(get_activation('final_pool'))

## Evaluate encoder embeddings for training data (all data)
encoder.eval()
if cuda:
    train_data = train_data.cuda()
t000 = time.time() # Measure run time of trained encoder to generate embedding
x_emb = encoder(train_data)
print(f'Time to generate embedding: {str(round((time.time()-t000)/60,6))} minutes')

## Create AnnData object for embedding for easy follow-up analyses and visualization
adata_embs = ad.AnnData(x_emb.cpu().detach().numpy())
adata_embs.obs[args.temporal_key] = df_label_comb[args.temporal_key].values
adata_embs.obs[f"{args.temporal_key}_codes"] = df_label_comb[f"{args.temporal_key}_codes"].values
adata_embs.obs[args.predictor_key] = df_label_comb[args.predictor_key].values

# ## Save embedding as AnnData object
# # X.csv is the model embedding (shape  = embedding dimension)
if args.save_model == True:
    res_dir = args.save_dir + 'embed'
    if not os.path.exists(res_dir): 
        os.makedirs(res_dir) 
    adata_embs.write_csvs(res_dir, skip_data = False, sep=",") 

# ==============================================================================
# =                              Visualization                                 =
# ==============================================================================

print("Visualizing model embedding...")

## Plot UMAP
plot_umap(adata_embs, args.predictor_key, predictor_labels, args.temporal_key, args.temporal_labels, seed = args.seed, save_dir = args.save_dir)

print("Process complete!")
