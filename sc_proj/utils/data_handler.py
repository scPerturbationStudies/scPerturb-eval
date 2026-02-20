import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
import pandas as pd
import os 
import random
from collections import defaultdict

# np.random.seed(42)

def prepare_train_data(adata, primary_variable, modality_variable, control_key, ood_primary, ood_modality, path_to_save, setting='ood', pid_percentage = 0.2):

    adata = adata[adata.obs[modality_variable].isin([ood_modality, control_key])].copy()

    global pid_suffix
    if pid_percentage is not None:
        pid_suffix = int(float(pid_percentage)*100)
    
    # Select the test data based on OOD criteria
    test_adata = adata[(adata.obs[primary_variable] == ood_primary) & (adata.obs[modality_variable] == ood_modality)].copy() 
    if test_adata.obs.index.empty:
        raise ValueError("No OOD indices found. Please check the input parameters.")
    
    # Initialize train_adata and valid_adata
    ood_indices = test_adata.obs.index
    non_ood_indices = adata.obs[~adata.obs.index.isin(ood_indices)].index
   
    if setting == 'pid':
        data = path_to_save.split('/')[-5]
        save_folder = path_to_save.split('/')[-3]
        final_folder = os.path.join('../outputs/dependencies', f'pid_test_indices/{data}/{pid_suffix}/{save_folder}/{ood_primary}-{ood_modality}')
        os.makedirs(final_folder, exist_ok=True)
        if os.path.isfile(os.path.join(final_folder,'test_indices.npy')):
            print(f"test indices file exists.")
            ood_indices_test = np.load(os.path.join(final_folder,'test_indices.npy'), allow_pickle=True)
        else:
            ood_indices_train, ood_indices_test = train_test_split(ood_indices, test_size= (1- pid_percentage))
            np.save(os.path.join(final_folder,'test_indices.npy'), ood_indices_test)        
            # indices = non_ood_indices.union(ood_indices_train)
    
        indices = adata.obs[~adata.obs.index.isin(ood_indices_test)].index
        train_adata = adata[indices].copy()
    
    elif setting == 'ood':
        
        train_adata = adata[non_ood_indices].copy()

    elif setting == 'iid':
       train_adata = adata 
    
    return train_adata

def split_train_valid(adata, train_percentage=0.85):
    
    np.random.seed(42)
    
    # Initialize train_adata and valid_adata
    indices = adata.obs.index
    split = np.random.choice(['train', 'valid'], size=len(indices), p=[train_percentage, 1 - train_percentage])
    train_set = [indices[i] for i in range(len(indices)) if split[i] == 'train']
    valid_set = [indices[i] for i in range(len(indices)) if split[i] == 'valid']
    
    train_adata = adata[train_set].copy()
    valid_adata = adata[valid_set].copy()
        
    return train_adata, valid_adata
     
class GPUDataset(Dataset):
    def __init__(self, adata, ordered_indices=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if 'onehots' in adata.obsm:
            if ordered_indices is not None:
                data = adata[ordered_indices].X
                onehot = adata[ordered_indices].obsm['onehots']
            else:
                data = adata.X
                onehot = adata.obsm['onehots']
            
            if issparse(data):
                data = data.toarray()
            
            self.data = torch.FloatTensor(data).to(device)
            self.onehot = torch.FloatTensor(onehot).to(device)
            self.has_onehot = True
        else:
            # Handle the case when 'onehots' is not present
            if ordered_indices is not None:
                data = adata[ordered_indices].X
            else:
                data = adata.X
            
            if issparse(data):
                data = data.toarray()
            
            self.data = torch.FloatTensor(data).to(device)
            self.has_onehot = False

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self._get_single_observation(index)
  
    def _get_single_observation(self, index):
        if self.has_onehot:
            return self.data[index], self.onehot[index]
        else:
            return self.data[index], torch.empty(0).to(self.data.device)

def prepare_data_loader(adata, indices = None, batch_size=512, shuffle=False):

    dataset = GPUDataset(adata, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


def shuffle_batch(data, condition, num_primaries):
    final_indices = []
    # indices = list(data.index)
    permutation_store_dict = defaultdict(list)
    for start in range(num_primaries):
        indx_for_permutation = np.arange(start, len(data), num_primaries)
        permutation_store_dict[start] = np.random.permutation(indx_for_permutation)

    repeats = len(permutation_store_dict[0])
    for r in range(repeats):
        for primary in range(len(permutation_store_dict.keys())):
            final_indices.append(permutation_store_dict[primary][r])

    # permutated_indices = [indices[n] for n in final_indices]
    data = data[final_indices]
    condition = condition[final_indices]
    
    return data, condition

        



    
