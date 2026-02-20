import os
import numpy as np
import pandas as pd
from utils.common import calculate_cpm, seurat_deg
from scipy.sparse import issparse


def predict_baseline_logfc(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save = '.'):
    if issparse(train_adata.X):
        train_adata.X = train_adata.X.toarray()
    print(setting)
    if setting == "ood":
        primaries = train_adata.obs[data_params['primary_variable']].unique().to_numpy()
        primaries = [x for x in primaries if x != ood_primary]
    else: 
        print("hello")
        primaries = train_adata.obs[data_params['primary_variable']].unique().to_numpy()
        
    print(primaries)
    logfcs = []
    gene_names = train_adata.var_names

    # non_ood_adata = train_adata[train_adata.obs[data_params['primary_variable']] != ood_primary].copy()
    non_ood_adata = train_adata[(train_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                (train_adata.obs[data_params['modality_variable']].isin([data_params['control_key'], ood_modality]))].copy()

    for primary in primaries:
        # adata = train_adata[train_adata.obs[data_params['primary_variable']] == primary].copy()
        adata = train_adata[(train_adata.obs[data_params['primary_variable']] == primary) & (train_adata.obs[data_params['modality_variable']].isin([data_params['control_key'], ood_modality]))].copy()
        seurat_df = seurat_deg(
            adata, data_params['modality_variable'], ood_modality, data_params['control_key'], \
            'MAST', find_variable_features=False, scale_factor=1e4)
        to_append = np.zeros(len(gene_names))
        for i,gene in enumerate(gene_names):
            if gene  in seurat_df.index:
                fraction = adata.obs.shape[0] / non_ood_adata.obs.shape[0]
                to_append[i] = seurat_df.loc[gene]['avg_log2FC'] * fraction
            else:
                to_append[i] = 0
        logfcs.append(to_append)
        # logfcs.append(seurat_df.loc[gene_names]['avg_log2FC'].values)
    
    logfcs_mean = pd.DataFrame(logfcs).sum(axis=0).values
    logfcs_mean = np.log(2)*logfcs_mean
    ood_adata = train_adata[(train_adata.obs[data_params['primary_variable']] == ood_primary) &\
        (train_adata.obs[data_params['modality_variable']] == data_params['control_key'])].copy()
    
    ood_adata.X = np.exp(logfcs_mean) * ood_adata.X
    ood_adata.X = np.where(ood_adata.X < 0, 0, ood_adata.X)

    ood_adata.layers['counts'] = ood_adata.X.copy()

    calculate_cpm(ood_adata)
    # ood_adata.X = ood_adata.X + logfcs_mean

    print(f'{ood_primary} control size: {ood_adata.obs.shape}')
     
    pred_adata = ood_adata.copy()
    pred_adata.obs[data_params['modality_variable']] = 'baseline_logfc_pred' 

    # total_count = 1e6
    # total_count = 1e4
    # pred_adata.layers['counts'] = (np.exp(np.asarray(pred_adata.X))-1)/total_count

    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    pred_adata.write(pred_file_path)
     

