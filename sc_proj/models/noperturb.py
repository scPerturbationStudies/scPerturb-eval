import os
from utils.common import calculate_cpm

class noperturb:
    def __init__(self, adata, data_params):
        self.data_params = data_params
        self.adata = adata
        self.primary_variable =  self.data_params['primary_variable']
        self.modality_variable =  self.data_params['modality_variable']
        self.ood_modality =  self.data_params['ood_modality']
        self.control_key = data_params['control_key']

    def forward(self, ood_primary):

        primary_subset = self.adata[self.adata.obs[self.primary_variable] == ood_primary]
        nopert_adata = primary_subset[primary_subset.obs[self.modality_variable] == self.control_key]
               
        return nopert_adata
    

def predict_noperturb(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save = '.'):

     train_adata = calculate_cpm(train_adata)
     
     # BUILD MODEL
     model = noperturb(train_adata, data_params)
     nopert_adata = model.forward(ood_primary)
     
     print(f'{ood_primary} control size: {nopert_adata.obs.shape}')
     
     pred_adata = nopert_adata.copy()
     pred_adata.obs[data_params['modality_variable']] = 'nopert_pred' 
     
     pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
     # Save the predicted AnnData object
     pred_adata.write(pred_file_path)
     

