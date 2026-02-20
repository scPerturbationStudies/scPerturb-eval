import os
import numpy as np
from utils.common import calculate_cpm

def predict_random_all(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save = '.'):

    train_adata = calculate_cpm(train_adata)

    all_stim = train_adata[train_adata.obs[data_params['modality_variable']]==ood_modality].copy()
    ctrl_primary = train_adata[(train_adata.obs[data_params['modality_variable']]==data_params['control_key']) &
                                (train_adata.obs[data_params['primary_variable']]==ood_primary)].copy()
    
    print(f'{ood_primary} control size: {ctrl_primary.n_obs}')

    sampled_adata = adata_sample(all_stim, ctrl_primary.n_obs)
    
    pred_adata = sampled_adata.copy()
    pred_adata.obs[data_params['modality_variable']] = 'random_all' 
    pred_adata.obs_names_make_unique()
    
    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    # Save the predicted AnnData object
    pred_adata.write(pred_file_path)
     

def adata_sample(adata, size):
    replace = False
    if adata.n_obs < size:
        replace = True
    chosen_indices = np.random.choice(adata.n_obs, size=size, replace=replace)
    chosen_adata = adata[chosen_indices].copy()

    return chosen_adata