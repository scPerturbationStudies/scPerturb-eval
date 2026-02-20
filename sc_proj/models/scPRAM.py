from scpram import models
import torch
import os
import scanpy as sc
import numpy as np
from utils.common import calculate_cpm



def train_scPRAM(train_adata, hyperparams, data_params, setting, ood_primary=None, path_to_save='.'):
    
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    count_per_cell = 1e4
    
    model = models.SCPRAM(input_dim=train_adata.n_vars, device=device)
    model = model.to(model.device)
    
    #####################################################################
    # cpm instead of normalize+log1p
    # train_adata = calculate_cpm(train_adata)
    sc.pp.normalize_per_cell(train_adata, counts_per_cell_after=count_per_cell)
    sc.pp.log1p(train_adata)
    # sc.pp.scale(train_adata, max_value=10)
    
    model.train_SCPRAM(train_adata, epochs=hyperparams['train_epochs'])
    model_save_path = os.path.join(path_to_save, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, model_save_path)
    

def predict_scPRAM(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    count_per_cell = 1e4
    
     #####################################################################
    # cpm instead of normalize+log1p
    # train_adata = calculate_cpm(train_adata)
    sc.pp.normalize_per_cell(train_adata, counts_per_cell_after=count_per_cell)
    sc.pp.log1p(train_adata)
    # sc.pp.scale(train_adata, max_value=10)
    
    model = models.SCPRAM(input_dim=train_adata.n_vars, device=device)
    model = model.to(model.device)
    
    model_path = os.path.join(path_to_save, 'final_model.pth')  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['model_state_dict'])
      
    key_dic = {'condition_key': data_params['modality_variable'],
               'cell_type_key': data_params['primary_variable'],
               'ctrl_key': data_params['control_key'],
               'stim_key': ood_modality,
               'pred_key': 'predict',
               }
    
    pred_adata = model.predict(train_adata=train_adata,
                         cell_to_pred=ood_primary,
                         key_dic=key_dic)
    
    print(pred_adata.shape)
    pred_adata.obs[data_params['modality_variable']] = 'scPRAM_pred' 

    # pred_adata.X = ((np.exp(np.asarray(pred_adata.X))-1)/count_per_cell) * 1e4
    # sc.pp.log1p(pred_adata)
    
    print("Prediction complete.")
    
    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    # Save the predicted AnnData object
    pred_adata.write(pred_file_path)
  
    
    
    